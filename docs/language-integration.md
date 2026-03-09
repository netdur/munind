# Language Integration

Munind can be integrated from non-Rust languages through `munind-cli` JSON commands.

This is the easiest cross-language path today for C, Python, Go, Node, etc.

## Build CLI Once

```bash
cargo build -p munind-cli --release
```

Binary path:

`target/release/munind-cli`

## CRUD Contract

- `insert` returns JSON: `{"status":"ok","id":<u64>}`
- `get` returns JSON with `id`, `document`, and optional `embedding`
- `update` returns JSON: `{"status":"ok","id":<u64>}`
- `delete` returns JSON: `{"status":"ok","id":<u64>}`
- `search --json` returns JSON array of hits

All commands accept `--db <path>`.

## Python Example

```python
import json
import subprocess

CLI = "target/release/munind-cli"
DB = "./munind_data"

def run(*args):
    out = subprocess.check_output([CLI, "--db", DB, *args], text=True)
    return json.loads(out)

# insert
created = run(
    "insert",
    "--embedding-json", "[0.1,0.2,0.3]",
    "--document-json", '{"doc_id":"p1","text":"hello from python"}',
)
rid = created["id"]

# get
record = run("get", "--id", str(rid), "--include-embedding", "--json")

# update
run(
    "update",
    "--id", str(rid),
    "--embedding-json", "[0.3,0.2,0.1]",
    "--document-json", '{"doc_id":"p1","text":"updated from python"}',
)

# delete
run("delete", "--id", str(rid))
```

## C Example (System Call)

```c
#include <stdlib.h>

int main(void) {
  system("target/release/munind-cli --db ./munind_data "
         "insert --embedding-json '[0.1,0.2,0.3]' "
         "--document-json '{\"doc_id\":\"c1\",\"text\":\"hello from c\"}'");

  system("target/release/munind-cli --db ./munind_data get --id 1 --json");
  system("target/release/munind-cli --db ./munind_data "
         "update --id 1 --embedding-json '[0.2,0.2,0.2]' "
         "--document-json '{\"doc_id\":\"c1\",\"text\":\"updated from c\"}'");
  system("target/release/munind-cli --db ./munind_data delete --id 1");
  return 0;
}
```

For production C integration, prefer `popen`/`exec` with structured stdout parsing.

## Notes

- IDs are numeric and stable until deleted.
- `update` keeps the same ID.
- No manual reindex step is needed after insert/update/delete while DB is open.
- For faster reopen after many writes, run:

```bash
target/release/munind-cli --db ./munind_data optimize --checkpoint-wal-only
```
