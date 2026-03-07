# 07 - API, CLI, And Observability

## Rust API (Library)
Primary API surface:
- `Engine::create(path, embedding_dimension, config)`
- `Engine::open(path, config)`
- `engine.insert_json(embedding, document_json)`
- `engine.insert_json_batch(rows)`
- `engine.search(request)`
- `engine.delete(id)`
- `engine.optimize(...)`
- `engine.snapshot()`

## Integration Surface
- Rust library is the primary interface.
- CLI is required.
- Optional local adapters (desktop/mobile) are allowed.
- Integration order: desktop first, then mobile.
- Cloud/server deployment is not a target scope for v1.

## CLI Commands (NGT Command Style, Rust Native)
- `munind create --config ... --embedding-dim ...`
- `munind insert --file ...`
- `munind search --query ... --top-k ... --where \"field=value\"`
- `munind remove --id ...`
- `munind stats`
- `munind optimize`
- `munind snapshot`
- `munind repair`
- `munind benchmark`

First profile default:
- Use `--embedding-dim 512` unless explicitly overridden.

## Config File
Use `TOML` with explicit sections:
- `[storage]`
- `[index]`
- `[query]`
- `[runtime]`
- `[telemetry]`

## Observability
Metrics (Prometheus/OpenTelemetry):
- query latency histogram
- insert throughput
- WAL fsync latency
- index graph degree distribution
- recall drift (periodic sampled)
- queue depths and background job durations

Tracing:
- structured spans for `search`, `insert`, `snapshot`, `recovery`.

Logging:
- JSON logs by default.
- redaction guard for sensitive JSON fields.

## Operations
- Startup checks include manifest/WAL sanity.
- CLI `munind stats` includes recovery status and index health.
