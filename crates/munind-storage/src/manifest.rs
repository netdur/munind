use std::path::{Path, PathBuf};
use std::fs;
use std::sync::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use munind_core::error::{MunindError, Result};
use munind_core::config::EngineConfig;
use munind_core::domain::{MemoryId, SearchRequest, SearchHit, OptimizeRequest, OptimizeReport};
use munind_core::engine::VectorEngine;

use crate::wal::{WalFile, WalRecord, OpType};
use crate::segment::{VectorSegment, JsonSegment};
use crate::id::IdAllocator;

#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub embedding_dimension: usize,
    pub config: EngineConfig,
}

/// Core engine orchestrating Storage in Phase 1
pub struct StorageEngine {
    _data_dir: PathBuf,
    pub manifest: Manifest,
    
    wal: Mutex<WalFile>,
    vec_seg: Mutex<VectorSegment>,
    json_seg: Mutex<JsonSegment>,
    pub id_alloc: RwLock<IdAllocator>,
}

impl StorageEngine {
    pub fn create<P: AsRef<Path>>(path: P, embedding_dimension: usize, config: EngineConfig) -> Result<Self> {
        let dir = path.as_ref();
        if embedding_dimension == 0 {
            return Err(MunindError::InvalidConfig(
                "embedding_dimension must be greater than zero".to_string(),
            ));
        }
        let manifest_path = dir.join("MANIFEST.json");
        if manifest_path.exists() {
            return Err(MunindError::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "Database manifest already exists in directory",
            )));
        }

        fs::create_dir_all(dir)?;
        fs::create_dir_all(dir.join("wal"))?;
        fs::create_dir_all(dir.join("segments"))?;
        fs::create_dir_all(dir.join("index"))?;
        fs::create_dir_all(dir.join("snapshots"))?;

        let manifest = Manifest {
            version: 1,
            embedding_dimension,
            config: config.clone(),
        };

        fs::write(
            dir.join("MANIFEST.json"),
            serde_json::to_string_pretty(&manifest)?,
        )?;

        let wal = WalFile::open(dir.join("wal/000001.wal"), config.storage.fsync_enabled)?;
        let vec_seg = VectorSegment::open(dir.join("segments/vectors-000001.seg"), embedding_dimension)?;
        let json_seg = JsonSegment::open(dir.join("segments/docs-000001.seg"))?;

        Ok(Self {
            _data_dir: dir.to_path_buf(),
            manifest,
            wal: Mutex::new(wal),
            vec_seg: Mutex::new(vec_seg),
            json_seg: Mutex::new(json_seg),
            id_alloc: RwLock::new(IdAllocator::new()),
        })
    }

    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let dir = path.as_ref();
        let manifest_path = dir.join("MANIFEST.json");

        if !manifest_path.exists() {
            return Err(MunindError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "MANIFEST.json not found",
            )));
        }

        let manifest: Manifest = serde_json::from_str(&fs::read_to_string(manifest_path)?)?;
        
        let mut wal = WalFile::open(dir.join("wal/000001.wal"), manifest.config.storage.fsync_enabled)?;
        let mut vec_seg = VectorSegment::open(dir.join("segments/vectors-000001.seg"), manifest.embedding_dimension)?;
        let mut json_seg = JsonSegment::open(dir.join("segments/docs-000001.seg"))?;
        let mut id_alloc = IdAllocator::new();

        // Rebuild segments + in-memory state from WAL to avoid replay-appending
        // into previously materialized segments on each open.
        vec_seg.reset()?;
        json_seg.reset()?;

        // Replay WAL
        wal.replay(|record| {
            match record.op {
                OpType::Insert { embedding, document } => {
                    let v_off = vec_seg.append(&embedding)?;
                    let j_off = json_seg.append(&document)?;
                    id_alloc.set_location(record.memory_id, v_off, j_off);
                }
                OpType::Update { embedding, document } => {
                    let v_off = vec_seg.append(&embedding)?;
                    let j_off = json_seg.append(&document)?;
                    id_alloc.set_location(record.memory_id, v_off, j_off);
                }
                OpType::Delete => {
                    id_alloc.tombstone(record.memory_id);
                }
                OpType::Config => {}
            }
            Ok(())
        })?;

        Ok(Self {
            _data_dir: dir.to_path_buf(),
            manifest,
            wal: Mutex::new(wal),
            vec_seg: Mutex::new(vec_seg),
            json_seg: Mutex::new(json_seg),
            id_alloc: RwLock::new(id_alloc),
        })
    }

    pub fn close(&self) -> Result<()> {
        let _wal = self.wal.lock().unwrap();
        // fsync logic could go here
        Ok(())
    }

    pub fn get_all_ids(&self) -> Vec<MemoryId> {
        let alloc = self.id_alloc.read().unwrap();
        let mut ids = Vec::new();
        // The id allocator tracks up to next_id.0. We expose a next_id() method on alloc 
        // to avoid mutating state just to know the bound.
        for i in 1..alloc.next_id().0 {
            let id = MemoryId(i);
            if alloc.get_location(id).is_some() {
                ids.push(id);
            }
        }
        ids
    }

    pub fn get_vector(&self, id: MemoryId) -> Result<Option<Vec<f32>>> {
        let alloc = self.id_alloc.read().unwrap();
        if let Some(loc) = alloc.get_location(id) {
            let mut seg = self.vec_seg.lock().unwrap();
            Ok(Some(seg.read(loc.vector_offset)?))
        } else {
            Ok(None)
        }
    }

    pub fn get_document(&self, id: MemoryId) -> Result<Option<Value>> {
        let alloc = self.id_alloc.read().unwrap();
        if let Some(loc) = alloc.get_location(id) {
            let mut seg = self.json_seg.lock().unwrap();
            let val = seg.read(loc.json_offset)?;
            Ok(Some(val))
        } else {
            Ok(None)
        }
    }
}

impl VectorEngine for StorageEngine {
    fn create_database(&self, _dim: usize, _cfg: EngineConfig) -> Result<()> {
        Err(MunindError::Internal("Use static create() instead".into()))
    }

    fn insert_json(&self, embedding: Vec<f32>, document: Value) -> Result<MemoryId> {
        if embedding.len() != self.manifest.embedding_dimension {
            return Err(MunindError::DimensionMismatch {
                expected: self.manifest.embedding_dimension,
                actual: embedding.len(),
            });
        }

        let mut alloc = self.id_alloc.write().unwrap();
        let id = alloc.allocate();

        let record = WalRecord {
            op: OpType::Insert { embedding: embedding.clone(), document: document.clone() },
            memory_id: id,
        };

        // 1. Write to WAL
        self.wal.lock().unwrap().append(&record)?;

        // 2. Write to segments
        let v_off = self.vec_seg.lock().unwrap().append(&embedding)?;
        let j_off = self.json_seg.lock().unwrap().append(&document)?;

        // 3. Update in-memory struct
        alloc.set_location(id, v_off, j_off);

        Ok(id)
    }

    fn insert_json_batch(&self, rows: Vec<(Vec<f32>, Value)>) -> Result<Vec<MemoryId>> {
        let mut ids = Vec::new();
        for (emb, doc) in rows {
            ids.push(self.insert_json(emb, doc)?);
        }
        Ok(ids)
    }

    fn search(&self, _query: SearchRequest) -> Result<Vec<SearchHit>> {
        unimplemented!("Graph Search implemention pending Phase 2")
    }

    fn remove(&self, id: MemoryId) -> Result<()> {
        let mut alloc = self.id_alloc.write().unwrap();

        if alloc.get_location(id).is_none() {
            return Err(MunindError::NotFound(id.0));
        }

        let record = WalRecord { op: OpType::Delete, memory_id: id };
        self.wal.lock().unwrap().append(&record)?;
        let removed = alloc.tombstone(id);
        debug_assert!(removed);

        Ok(())
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }

    fn optimize(&self, _req: OptimizeRequest) -> Result<OptimizeReport> {
        Ok(OptimizeReport::default())
    }
}
