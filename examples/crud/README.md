# CRUD Example (Rust)

This folder is a standalone example project for Munind CRUD operations.

Files:
- `src/main.rs`: end-to-end flow runner (create -> read -> update -> read -> delete)
- `src/create.rs`: create/insert operation
- `src/read.rs`: multiple read examples (get, vector search, text search, hybrid, folder filter)
- `src/update.rs`: update operation (same ID, new embedding/document)
- `src/delete.rs`: delete operation + existence check

## Run

From repository root:

```bash
cargo run --manifest-path examples/crud/Cargo.toml
```

Optional custom DB path:

```bash
MUNIND_EXAMPLE_DB=/tmp/munind_crud_example \
  cargo run --manifest-path examples/crud/Cargo.toml
```

The example always deletes the target DB path before running, so it is repeatable.
