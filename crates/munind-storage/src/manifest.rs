use crate::id::IdAllocator;
use crate::segment::{JsonSegment, VectorSegment};
use crate::wal::{OpType, WalFile, WalRecord};
use munind_core::config::EngineConfig;
use munind_core::domain::{MemoryId, OptimizeReport, OptimizeRequest, SearchHit, SearchRequest};
use munind_core::engine::VectorEngine;
use munind_core::error::{MunindError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub embedding_dimension: usize,
    pub config: EngineConfig,
}

const CHECKPOINT_FILE: &str = "snapshots/state-checkpoint.json";
const CHECKPOINT_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
struct StorageCheckpoint {
    version: u32,
    id_allocator: IdAllocator,
}

/// Core engine orchestrating durable storage.
pub struct StorageEngine {
    data_dir: PathBuf,
    pub manifest: Manifest,

    wal: Mutex<WalFile>,
    vec_seg: Mutex<VectorSegment>,
    json_seg: Mutex<JsonSegment>,
    pub id_alloc: RwLock<IdAllocator>,
}

impl StorageEngine {
    pub fn create<P: AsRef<Path>>(
        path: P,
        embedding_dimension: usize,
        config: EngineConfig,
    ) -> Result<Self> {
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
        let vec_seg =
            VectorSegment::open(dir.join("segments/vectors-000001.seg"), embedding_dimension)?;
        let json_seg = JsonSegment::open(dir.join("segments/docs-000001.seg"))?;

        Ok(Self {
            data_dir: dir.to_path_buf(),
            manifest,
            wal: Mutex::new(wal),
            vec_seg: Mutex::new(vec_seg),
            json_seg: Mutex::new(json_seg),
            id_alloc: RwLock::new(IdAllocator::new()),
        })
    }

    fn checkpoint_path(dir: &Path) -> PathBuf {
        dir.join(CHECKPOINT_FILE)
    }

    fn load_checkpoint(dir: &Path) -> Result<Option<IdAllocator>> {
        let path = Self::checkpoint_path(dir);
        if !path.exists() {
            return Ok(None);
        }

        let checkpoint: StorageCheckpoint = serde_json::from_str(&fs::read_to_string(&path)?)?;
        if checkpoint.version != CHECKPOINT_VERSION {
            return Err(MunindError::Corruption(format!(
                "Unsupported checkpoint version: {}",
                checkpoint.version
            )));
        }

        Ok(Some(checkpoint.id_allocator))
    }

    fn write_checkpoint_atomic(dir: &Path, id_allocator: &IdAllocator) -> Result<()> {
        let checkpoint = StorageCheckpoint {
            version: CHECKPOINT_VERSION,
            id_allocator: id_allocator.clone(),
        };

        let checkpoint_path = Self::checkpoint_path(dir);
        let checkpoint_tmp = checkpoint_path.with_extension("tmp");

        fs::write(&checkpoint_tmp, serde_json::to_vec_pretty(&checkpoint)?)?;
        fs::rename(&checkpoint_tmp, &checkpoint_path)?;
        Ok(())
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

        let mut wal = WalFile::open(
            dir.join("wal/000001.wal"),
            manifest.config.storage.fsync_enabled,
        )?;
        let mut vec_seg = VectorSegment::open(
            dir.join("segments/vectors-000001.seg"),
            manifest.embedding_dimension,
        )?;
        let mut json_seg = JsonSegment::open(dir.join("segments/docs-000001.seg"))?;
        let checkpoint_alloc = Self::load_checkpoint(dir)?;
        let mut id_alloc = checkpoint_alloc.clone().unwrap_or_default();

        // If no checkpoint exists, rebuild materialized segments from scratch by replaying WAL.
        // If a checkpoint exists, keep segments as-is and replay only WAL tail since checkpoint.
        if checkpoint_alloc.is_none() {
            vec_seg.reset()?;
            json_seg.reset()?;
        }
        wal.replay(|record| {
            match record.op {
                OpType::Insert {
                    embedding,
                    document,
                }
                | OpType::Update {
                    embedding,
                    document,
                } => {
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
            data_dir: dir.to_path_buf(),
            manifest,
            wal: Mutex::new(wal),
            vec_seg: Mutex::new(vec_seg),
            json_seg: Mutex::new(json_seg),
            id_alloc: RwLock::new(id_alloc),
        })
    }

    fn wal_lock(&self) -> Result<MutexGuard<'_, WalFile>> {
        self.wal
            .lock()
            .map_err(|_| MunindError::Internal("wal lock poisoned".to_string()))
    }

    fn vec_lock(&self) -> Result<MutexGuard<'_, VectorSegment>> {
        self.vec_seg
            .lock()
            .map_err(|_| MunindError::Internal("vector segment lock poisoned".to_string()))
    }

    fn json_lock(&self) -> Result<MutexGuard<'_, JsonSegment>> {
        self.json_seg
            .lock()
            .map_err(|_| MunindError::Internal("json segment lock poisoned".to_string()))
    }

    fn id_read(&self) -> Result<RwLockReadGuard<'_, IdAllocator>> {
        self.id_alloc
            .read()
            .map_err(|_| MunindError::Internal("id allocator read lock poisoned".to_string()))
    }

    fn id_write(&self) -> Result<RwLockWriteGuard<'_, IdAllocator>> {
        self.id_alloc
            .write()
            .map_err(|_| MunindError::Internal("id allocator write lock poisoned".to_string()))
    }

    pub fn close(&self) -> Result<()> {
        self.sync_all()
    }

    pub fn sync_all(&self) -> Result<()> {
        self.wal_lock()?.flush()?;
        self.vec_lock()?.flush()?;
        self.json_lock()?.flush()?;
        Ok(())
    }

    fn storage_bytes(&self) -> u64 {
        let wal = std::fs::metadata(self.data_dir.join("wal/000001.wal"))
            .map(|m| m.len())
            .unwrap_or(0);
        let vec = std::fs::metadata(self.data_dir.join("segments/vectors-000001.seg"))
            .map(|m| m.len())
            .unwrap_or(0);
        let doc = std::fs::metadata(self.data_dir.join("segments/docs-000001.seg"))
            .map(|m| m.len())
            .unwrap_or(0);
        wal + vec + doc
    }

    pub fn get_all_ids(&self) -> Result<Vec<MemoryId>> {
        let alloc = self.id_read()?;

        let mut ids = Vec::new();
        for i in 1..alloc.next_id().0 {
            let id = MemoryId(i);
            if alloc.get_location(id).is_some() {
                ids.push(id);
            }
        }
        Ok(ids)
    }

    pub fn get_vector(&self, id: MemoryId) -> Result<Option<Vec<f32>>> {
        let loc = {
            let alloc = self.id_read()?;
            alloc.get_location(id).cloned()
        };

        match loc {
            Some(loc) => {
                let mut seg = self.vec_lock()?;
                Ok(Some(seg.read(loc.vector_offset)?))
            }
            None => Ok(None),
        }
    }

    pub fn get_document(&self, id: MemoryId) -> Result<Option<Value>> {
        let loc = {
            let alloc = self.id_read()?;
            alloc.get_location(id).cloned()
        };

        match loc {
            Some(loc) => {
                let mut seg = self.json_lock()?;
                Ok(Some(seg.read(loc.json_offset)?))
            }
            None => Ok(None),
        }
    }

    pub fn update_json(&self, id: MemoryId, embedding: Vec<f32>, document: Value) -> Result<()> {
        if embedding.len() != self.manifest.embedding_dimension {
            return Err(MunindError::DimensionMismatch {
                expected: self.manifest.embedding_dimension,
                actual: embedding.len(),
            });
        }

        let exists = {
            let alloc = self.id_read()?;
            alloc.get_location(id).is_some()
        };
        if !exists {
            return Err(MunindError::NotFound(id.0));
        }

        let record = WalRecord {
            op: OpType::Update {
                embedding: embedding.clone(),
                document: document.clone(),
            },
            memory_id: id,
        };
        self.wal_lock()?.append(&record)?;

        let v_off = self.vec_lock()?.append(&embedding)?;
        let j_off = self.json_lock()?.append(&document)?;

        let mut alloc = self.id_write()?;
        alloc.set_location(id, v_off, j_off);
        Ok(())
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

        let id = {
            let mut alloc = self.id_write()?;
            alloc.allocate()
        };

        let record = WalRecord {
            op: OpType::Insert {
                embedding: embedding.clone(),
                document: document.clone(),
            },
            memory_id: id,
        };

        // Write-ahead log first for durability.
        self.wal_lock()?.append(&record)?;

        // Then materialize into segments.
        let v_off = self.vec_lock()?.append(&embedding)?;
        let j_off = self.json_lock()?.append(&document)?;

        let mut alloc = self.id_write()?;
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
        Err(MunindError::Internal(
            "StorageEngine does not provide search; use MunindEngine".to_string(),
        ))
    }

    fn remove(&self, id: MemoryId) -> Result<()> {
        let exists = {
            let alloc = self.id_read()?;
            alloc.get_location(id).is_some()
        };

        if !exists {
            return Err(MunindError::NotFound(id.0));
        }

        let record = WalRecord {
            op: OpType::Delete,
            memory_id: id,
        };
        self.wal_lock()?.append(&record)?;

        let mut alloc = self.id_write()?;
        let removed = alloc.tombstone(id);
        if !removed {
            return Err(MunindError::NotFound(id.0));
        }

        Ok(())
    }

    fn flush(&self) -> Result<()> {
        self.sync_all()
    }

    fn optimize(&self, req: OptimizeRequest) -> Result<OptimizeReport> {
        if req.checkpoint_wal_only && !req.force_full_compaction {
            let before_bytes = self.storage_bytes();

            let mut wal = self.wal_lock()?;
            let mut vec_seg = self.vec_lock()?;
            let mut json_seg = self.json_lock()?;
            let alloc = self.id_read()?.clone();

            // Persist current materialized state and allocator snapshot, then clear WAL tail.
            vec_seg.flush()?;
            json_seg.flush()?;
            Self::write_checkpoint_atomic(&self.data_dir, &alloc)?;
            wal.reset()?;

            drop(json_seg);
            drop(vec_seg);
            drop(wal);
            self.sync_all()?;

            let after_bytes = self.storage_bytes();
            return Ok(OptimizeReport {
                records_compacted: 0,
                space_reclaimed_bytes: before_bytes.saturating_sub(after_bytes),
                graph_edges_repaired: 0,
            });
        }

        if !req.force_full_compaction {
            return Ok(OptimizeReport::default());
        }

        let before_bytes = self.storage_bytes();

        let ids = self.get_all_ids()?;
        let mut rows = Vec::with_capacity(ids.len());
        for id in ids {
            if let (Some(embedding), Some(document)) =
                (self.get_vector(id)?, self.get_document(id)?)
            {
                rows.push((id, embedding, document));
            }
        }

        let mut wal = self.wal_lock()?;
        let mut vec_seg = self.vec_lock()?;
        let mut json_seg = self.json_lock()?;
        let mut alloc = self.id_write()?;

        vec_seg.reset()?;
        json_seg.reset()?;

        let mut compacted_alloc = IdAllocator::new();
        for (id, embedding, document) in rows.iter() {
            let v_off = vec_seg.append(embedding)?;
            let j_off = json_seg.append(document)?;
            compacted_alloc.set_location(*id, v_off, j_off);
        }

        *alloc = compacted_alloc.clone();

        // Persist compacted segments + allocator checkpoint before WAL truncation.
        vec_seg.flush()?;
        json_seg.flush()?;
        Self::write_checkpoint_atomic(&self.data_dir, &compacted_alloc)?;

        // WAL now only needs to contain post-checkpoint mutations.
        wal.reset()?;

        drop(alloc);
        drop(json_seg);
        drop(vec_seg);
        drop(wal);

        self.sync_all()?;
        let after_bytes = self.storage_bytes();

        Ok(OptimizeReport {
            records_compacted: rows.len(),
            space_reclaimed_bytes: before_bytes.saturating_sub(after_bytes),
            graph_edges_repaired: 0,
        })
    }
}
