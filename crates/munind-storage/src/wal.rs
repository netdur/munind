use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::Path;
use crc32fast::Hasher;
use serde::{Deserialize, Serialize};

use munind_core::error::{MunindError, Result};
use munind_core::domain::MemoryId;

pub const WAL_MAGIC: u32 = 0x4D554E41; // "MUNA"
pub const WAL_VERSION: u16 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpType {
    Insert {
        embedding: Vec<f32>,
        document: serde_json::Value,
    },
    Update {
        embedding: Vec<f32>,
        document: serde_json::Value,
    },
    Delete,
    Config, // Reserved for future
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalRecord {
    pub op: OpType,
    pub memory_id: MemoryId,
}

pub struct WalFile {
    file: File,
    fsync_enabled: bool,
}

impl WalFile {
    pub fn open<P: AsRef<Path>>(path: P, fsync_enabled: bool) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self { file, fsync_enabled })
    }

    pub fn append(&mut self, record: &WalRecord) -> Result<()> {
        let payload = serde_json::to_vec(record)
            .map_err(|e| MunindError::Internal(format!("Failed to serialize record: {}", e)))?;

        let mut hasher = Hasher::new();
        hasher.update(&WAL_MAGIC.to_le_bytes());
        hasher.update(&WAL_VERSION.to_le_bytes());
        let payload_len = payload.len() as u32;
        hasher.update(&payload_len.to_le_bytes());
        hasher.update(&payload);
        
        let crc = hasher.finalize();

        let mut buffer = Vec::with_capacity(4 + 2 + 4 + payload.len() + 4);
        buffer.extend_from_slice(&WAL_MAGIC.to_le_bytes());
        buffer.extend_from_slice(&WAL_VERSION.to_le_bytes());
        buffer.extend_from_slice(&payload_len.to_le_bytes());
        buffer.extend_from_slice(&payload);
        buffer.extend_from_slice(&crc.to_le_bytes());

        self.file.write_all(&buffer)?;
        
        if self.fsync_enabled {
            self.file.sync_data()?;
        }

        Ok(())
    }

    pub fn replay<F>(&mut self, mut callback: F) -> Result<()> 
    where 
        F: FnMut(WalRecord) -> Result<()>
    {
        self.file.seek(SeekFrom::Start(0))?;
        let mut reader = std::io::BufReader::new(&mut self.file);

        loop {
            let mut header = [0u8; 10]; // 4 (magic) + 2 (version) + 4 (len)
            match reader.read_exact(&mut header) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(MunindError::Io(e)),
            }

            let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
            if magic != WAL_MAGIC {
                return Err(MunindError::Corruption(format!("Bad WAL magic: {:x}", magic)));
            }

            let version = u16::from_le_bytes(header[4..6].try_into().unwrap());
            if version != WAL_VERSION {
                return Err(MunindError::Corruption(format!("Unknown WAL version: {}", version)));
            }

            let payload_len = u32::from_le_bytes(header[6..10].try_into().unwrap()) as usize;
            
            let mut payload = vec![0u8; payload_len];
            reader.read_exact(&mut payload)?;

            let mut crc_buf = [0u8; 4];
            reader.read_exact(&mut crc_buf)?;
            let expected_crc = u32::from_le_bytes(crc_buf);

            let mut hasher = Hasher::new();
            hasher.update(&header);
            hasher.update(&payload);
            let calculated_crc = hasher.finalize();

            if calculated_crc != expected_crc {
                return Err(MunindError::Corruption(format!("WAL checksum mismatch: expected {:x}, got {:x}", expected_crc, calculated_crc)));
            }

            let record: WalRecord = serde_json::from_slice(&payload)
                .map_err(|e| MunindError::Corruption(format!("Failed to deserialize payload: {}", e)))?;
            
            callback(record)?;
        }

        Ok(())
    }
}
