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
