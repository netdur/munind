pub mod manifest;
pub mod wal;
pub mod segment;
pub mod id;

pub use manifest::{StorageEngine, Manifest};
pub use wal::{WalFile, WalRecord, OpType};
pub use segment::{VectorSegment, JsonSegment};
pub use id::{IdAllocator, RecordLocation};

#[cfg(test)]
mod tests {
    use super::*;
    use munind_core::config::EngineConfig;
    use munind_core::domain::MemoryId;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_engine_create_open_close() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("mydb");
        let config = EngineConfig::default();

        // 1. Create the engine
        let engine = StorageEngine::create(&path, 512, config).expect("Failed to create engine");
        
        // Ensure manifest logic is sound
        assert!(path.join("MANIFEST.json").exists());
        assert!(path.join("wal").exists());
        assert!(path.join("segments").exists());
        assert!(path.join("index").exists());
        assert!(path.join("snapshots").exists());

        // Close it (dummy for Phase 0)
        engine.close().expect("Failed to close engine");

        // 2. Open the existing engine
        let opened_engine = StorageEngine::open(&path).expect("Failed to open engine");
        
        // It should open successfully
        opened_engine.close().expect("Failed to close opened engine");
    }

    #[test]
    fn test_wal_append_replay() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");

        let record1 = WalRecord {
            op: OpType::Insert {
                embedding: vec![1.0, 2.0, 3.0],
                document: serde_json::json!({"test": "doc1"}),
            },
            memory_id: MemoryId(1),
        };

        let record2 = WalRecord {
            op: OpType::Delete,
            memory_id: MemoryId(2),
        };

        // Append to WAL
        {
            let mut wal = WalFile::open(&path, true).unwrap();
            wal.append(&record1).unwrap();
            wal.append(&record2).unwrap();
        }

        // Replay from WAL
        {
            let mut wal = WalFile::open(&path, false).unwrap();
            let mut replayed = Vec::new();
            
            wal.replay(|rec| {
                replayed.push(rec);
                Ok(())
            }).unwrap();

            assert_eq!(replayed.len(), 2);
            assert_eq!(replayed[0].memory_id.0, 1);
            assert_eq!(replayed[1].memory_id.0, 2);
            
            match &replayed[0].op {
                OpType::Insert { embedding, .. } => {
                    assert_eq!(embedding.len(), 3);
                }
                _ => panic!("Expected Insert"),
            }
        }
    }

    #[test]
    fn test_segment_append_read() {
        let dir = tempdir().unwrap();
        
        let mut vec_seg = VectorSegment::open(dir.path().join("vec.seg"), 3).unwrap();
        let off1 = vec_seg.append(&[1.0, 2.0, 3.0]).unwrap();
        let off2 = vec_seg.append(&[4.0, 5.0, 6.0]).unwrap();
        
        assert_eq!(vec_seg.read(off1).unwrap(), vec![1.0, 2.0, 3.0]);
        assert_eq!(vec_seg.read(off2).unwrap(), vec![4.0, 5.0, 6.0]);

        let mut json_seg = JsonSegment::open(dir.path().join("doc.seg")).unwrap();
        let j_off1 = json_seg.append(&serde_json::json!({"id": 1})).unwrap();
        let j_off2 = json_seg.append(&serde_json::json!({"id": 2})).unwrap();

        assert_eq!(json_seg.read(j_off1).unwrap(), serde_json::json!({"id": 1}));
        assert_eq!(json_seg.read(j_off2).unwrap(), serde_json::json!({"id": 2}));
    }

    #[test]
    fn test_id_allocator() {
        let mut alloc = IdAllocator::new();
        let id1 = alloc.allocate();
        let id2 = alloc.allocate();
        assert_eq!(id1.0, 1);
        assert_eq!(id2.0, 2);

        alloc.set_location(id1, 10, 20);
        alloc.set_location(id2, 100, 200);

        assert_eq!(alloc.len(), 2);
        
        let loc1 = alloc.get_location(id1).unwrap();
        assert_eq!(loc1.vector_offset, 10);
        assert_eq!(loc1.json_offset, 20);

        assert!(alloc.tombstone(id1));
        assert!(alloc.get_location(id1).is_none());
        assert_eq!(alloc.len(), 1);
    }

    #[test]
    fn test_storage_engine_crash_restart() {
        use munind_core::engine::VectorEngine;
        
        let dir = tempdir().unwrap();
        let path = dir.path().join("db");
        let config = EngineConfig::default();

        let id1;
        let id2;
        let id3;

        // Simulate first run
        {
            let engine = StorageEngine::create(&path, 3, config).unwrap();
            
            id1 = engine.insert_json(
                vec![1.0, 1.0, 1.0], 
                serde_json::json!({"name": "doc1"})
            ).unwrap();
            
            id2 = engine.insert_json(
                vec![2.0, 2.0, 2.0], 
                serde_json::json!({"name": "doc2"})
            ).unwrap();
            
            id3 = engine.insert_json(
                vec![3.0, 3.0, 3.0], 
                serde_json::json!({"name": "doc3"})
            ).unwrap();

            engine.remove(id2).unwrap();
            // Engine drops here, simulating a crash (no flush/optimize graceful shutdown)
        }

        // Simulate restart
        {
            let engine = StorageEngine::open(&path).unwrap();
            let state = engine.id_alloc.read().unwrap();

            assert_eq!(state.len(), 2);
            assert!(state.get_location(id1).is_some());
            assert!(state.get_location(id2).is_none()); // tombstoned
            assert!(state.get_location(id3).is_some());

            // Since we know order, vectors should be exactly written
            let loc3 = state.get_location(id3).unwrap();
            assert!(loc3.vector_offset > 0);
            assert!(loc3.json_offset > 0);
        }
    }

    #[test]
    fn test_open_does_not_duplicate_segments() {
        use munind_core::engine::VectorEngine;

        let dir = tempdir().unwrap();
        let path = dir.path().join("db");
        let config = EngineConfig::default();

        {
            let engine = StorageEngine::create(&path, 3, config).unwrap();
            engine
                .insert_json(vec![1.0, 2.0, 3.0], serde_json::json!({"name": "a"}))
                .unwrap();
            engine
                .insert_json(vec![4.0, 5.0, 6.0], serde_json::json!({"name": "b"}))
                .unwrap();
        }

        let vec_path = path.join("segments/vectors-000001.seg");
        let doc_path = path.join("segments/docs-000001.seg");
        let vec_len_before = fs::metadata(&vec_path).unwrap().len();
        let doc_len_before = fs::metadata(&doc_path).unwrap().len();

        for _ in 0..3 {
            let engine = StorageEngine::open(&path).unwrap();
            engine.close().unwrap();
        }

        let vec_len_after = fs::metadata(&vec_path).unwrap().len();
        let doc_len_after = fs::metadata(&doc_path).unwrap().len();
        assert_eq!(vec_len_before, vec_len_after);
        assert_eq!(doc_len_before, doc_len_after);
    }

    #[test]
    fn test_remove_missing_returns_not_found() {
        use munind_core::engine::VectorEngine;
        use munind_core::error::MunindError;

        let dir = tempdir().unwrap();
        let path = dir.path().join("db");
        let config = EngineConfig::default();
        let engine = StorageEngine::create(&path, 3, config).unwrap();

        let err = engine.remove(MemoryId(999)).unwrap_err();
        assert!(matches!(err, MunindError::NotFound(999)));
    }
}
