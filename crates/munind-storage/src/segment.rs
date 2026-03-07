use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::Path;

use munind_core::error::{MunindError, Result};

/// A segment file that stores dense vectors.
pub struct VectorSegment {
    file: File,
    dimension: usize,
}

impl VectorSegment {
    pub fn open<P: AsRef<Path>>(path: P, dimension: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        Ok(Self { file, dimension })
    }

    /// Clears the segment contents and rewinds to start.
    pub fn reset(&mut self) -> Result<()> {
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        Ok(())
    }

    /// Appends a vector and returns the byte offset
    pub fn append(&mut self, vector: &[f32]) -> Result<u64> {
        if vector.len() != self.dimension {
            return Err(MunindError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.len(),
            });
        }
        
        let offset = self.file.seek(SeekFrom::End(0))?;
        
        // Convert f32 array to bytes
        // In a real optimized system, this would be unsafe transumting or bytemuck.
        let mut bytes = Vec::with_capacity(vector.len() * 4);
        for &v in vector {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        
        self.file.write_all(&bytes)?;
        Ok(offset)
    }

    pub fn read(&mut self, offset: u64) -> Result<Vec<f32>> {
        self.file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0u8; self.dimension * 4];
        self.file.read_exact(&mut bytes)?;
        
        let mut vector = Vec::with_capacity(self.dimension);
        for chunk in bytes.chunks_exact(4) {
            vector.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        Ok(vector)
    }
}

/// A segment file that stores JSON payloads.
pub struct JsonSegment {
    file: File,
}

impl JsonSegment {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        Ok(Self { file })
    }

    /// Clears the segment contents and rewinds to start.
    pub fn reset(&mut self) -> Result<()> {
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        Ok(())
    }

    /// Appends a JSON document and returns the byte offset
    pub fn append(&mut self, doc: &serde_json::Value) -> Result<u64> {
        let offset = self.file.seek(SeekFrom::End(0))?;
        let bytes = serde_json::to_vec(doc)?;
        
        // Write len then payload
        let len = bytes.len() as u32;
        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&bytes)?;
        
        Ok(offset)
    }

    pub fn read(&mut self, offset: u64) -> Result<serde_json::Value> {
        self.file.seek(SeekFrom::Start(offset))?;
        
        let mut len_buf = [0u8; 4];
        self.file.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;
        
        let mut bytes = vec![0u8; len];
        self.file.read_exact(&mut bytes)?;
        
        let doc = serde_json::from_slice(&bytes)?;
        Ok(doc)
    }
}
