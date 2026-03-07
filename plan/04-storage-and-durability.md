# 04 - Storage And Durability

## Disk Layout
```text
<data_root>/
  MANIFEST.json
  LOCK
  wal/
    00000001.wal
    00000002.wal
  segments/
    vectors-000001.seg
    docs-000001.seg
    filters-000001.seg
  index/
    graph-000001.idx
    seeds-000001.idx
  snapshots/
    snap-000001/
      MANIFEST.snapshot.json
      ...
```

## WAL Format
Each WAL record:
- `magic`
- `version`
- `op_type` (`insert`, `update`, `delete`, `config`)
- `memory_id`
- `embedding + json_document` bytes
- `crc32c`

Rules:
- Append before acknowledgment.
- Batched fsync based on durability mode.
- Reject corrupted tail on recovery after checksum boundary.

## Write Path
1. Validate record and vector dimensions.
2. Verify embedding dimension equals database creation-time dimension.
3. Append WAL record.
4. Write vector/JSON document to active segment.
5. Update in-memory ID map.
6. Mutate graph index (or enqueue if async mode).
7. Return ack.

## Recovery Path
1. Load latest manifest.
2. Open latest snapshot if present.
3. Replay WAL files in order.
4. Rebuild in-memory maps.
5. Run graph consistency quick-check.

## Snapshot Strategy
- Trigger by interval or WAL size threshold.
- Snapshot writes to temp folder then atomic rename.
- Keep N recent snapshots + WAL retention window.

## Compaction
- Repack live (non-tombstoned) records into new segments.
- Rebuild or incrementally repair graph references.
- Swap manifest atomically.

## Corruption Handling
- Checksums on WAL and index pages.
- Magic/version header for every binary file.
- Start in read-only degraded mode if hard corruption detected.
- Offer `munind-cli repair` to recover valid subset.

## Memory Mapping Policy
- Mmap read-mostly files (vector segments, graph pages).
- Keep WAL as sequential file IO.
- Avoid mmap for high-churn files to reduce fragmentation issues.

## Compatibility/Migrations
- Semantic format versioning in manifest.
- Migration code path per major/minor version.
- Block startup if major version unsupported.
