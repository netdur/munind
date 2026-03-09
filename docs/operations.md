# Operations

## Local Data Layout

A database directory contains:
- `MANIFEST.json`
- `wal/`
- `segments/`
- `index/`
- `snapshots/`

Example:

```bash
./munind_data/
```

## Backup and Restore

### Backup

For safest backup, stop writes first, then copy directory:

```bash
cp -R ./munind_data ./munind_data.backup
```

### Restore

```bash
rm -rf ./munind_data
cp -R ./munind_data.backup ./munind_data
```

## Health Check

```bash
cargo run -p munind-cli -- --db ./munind_data check-health
```

## Compaction and Rebuild

Programmatic optimize can compact storage and rebuild graph/index state:

- `force_full_compaction: true` rewrites live records.
- `repair_graph: true` rebuilds graph edges from current data.

CLI equivalent:

```bash
cargo run -p munind-cli -- --db ./munind_data optimize
```

## Common Issues

### 1) Dimension mismatch

Cause:
- Query/insert vector length differs from DB embedding dimension.

Fix:
- Ensure your embedding provider outputs exactly the DB dimension.
- Recreate DB if you need a different dimension.

### 2) Empty or weak search results

Checks:
- Verify embeddings are from the same model family as indexed data.
- Increase `ef_search`.
- Use hybrid (`text_query`) and optional reranking.
- Confirm filters match indexed fields and exact values.

### 3) Filter unexpectedly returns no results

Checks:
- `Eq` is exact JSON value match (type-sensitive).
- Path must match stored JSON (`source` vs `metadata.source`).

### 4) WAL file grows large

Cause:
- WAL keeps accumulating write records until a compaction/checkpoint pass runs.

Fix:
- Run:
  `cargo run -p munind-cli -- --db ./munind_data optimize`
- For benchmark flow, keep phase-2 compaction enabled (`COMPACT_AFTER_PREPARE=1`, default).

## Upgrade Safety

When introducing major index/search changes:
- Keep benchmark quality JSON snapshots in `benchmark/results/`.
- Re-run quality + latency benchmark before/after change.
- Compare recall@k and filtered recall@k, not only latency.
