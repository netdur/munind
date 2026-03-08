# End-State Expectations And Discussion Space

## Purpose
This file defines what the project should deliver at the end and is the shared space to align expectations as we build.

## Final Outcome (What You Will Get)
At project completion, you should have:
- A Rust local-only, general-purpose vector database.
- A local-only engine (single device), with no cloud scaling objective.
- A product shape that fits desktop and mobile usage better than server deployment.
- Persistent storage for embeddings and JSON documents.
- ANN search (NGT-style graph-centric design, based on `vendor/NGT/lib/NGT`, not `NGTQ`).
- Database-style workflows:
  - insert: embedding + JSON document
  - search: returns matching JSON documents (with ids and scores)
- Durable recovery path (WAL + snapshot + restore).
- CLI for common operations and maintenance.
- Benchmark and test coverage with measurable recall/latency baselines.
- Documentation/runbooks sufficient to operate and evolve the system.

Primary use cases include personal memory retrieval and local RAG context retrieval.

## Out Of Scope For Initial Delivery
- Distributed cluster mode.
- Cloud-native scaling architecture.
- Full NGTQ/QBG-style quantized architecture.
- Billion-scale guarantees.
- GPU-native indexing/search pipeline.

## Acceptance Criteria
Project is considered successful when all are true:
- Core insert/search/delete APIs are stable and tested.
- Embedding dimension is set at database creation and enforced for all writes.
- Insert accepts embedding + JSON document payload.
- Search returns JSON documents with scoring metadata.
- Vector search supports JSON equality filtering in v1 (`x = y` constraints).
- Crash recovery tests pass consistently.
- Search p95 target is <= 20 ms on agreed local benchmark profile.
- Recall and latency targets are met on agreed benchmark datasets.
- Data format/version handling is documented.
- Major operational tasks are runnable from CLI and documented.

## Target Quality Bar
- Correctness first, then optimization.
- Reproducible benchmarks.
- Clear error handling and observability.
- No silent data loss for acknowledged writes.
- Clear and predictable DB-like behavior.

## Assumptions
- Single-node, local-only deployment for v1.
- Primary target environments are desktop and mobile.
- Desktop integration is prioritized before mobile integration.
- Embeddings are provided externally (pluggable source).
- Embedding dimension is immutable per database instance after `create`.
- Default embedding dimension for the first database profile is `512`.

## Discussion Log (Use During Planning)
Add notes here during expectation discussions.

### Resolved Inputs
- [x] Default embedding dimension for first database profile: `512`.
- [x] Hard p95 latency target for initial milestone: `20 ms`.
- [x] Required JSON filter operator in v1: vector search constrained by JSON equality (`x = y`).
- [x] Integration priority: desktop first, then mobile.

### Decisions
| Date | Decision | Why | Impact |
|---|---|---|---|
| 2026-03-07 | Created expectation baseline doc | Align scope and final outcome | Shared reference for future changes |
| 2026-03-07 | Project is local-only, no cloud scaling target | Match intended use | Architecture and roadmap avoid distributed/server scope |
| 2026-03-07 | Embedding dimension is fixed at database creation | Keep storage/index invariants simple and strong | Write path validates strict dimension match |
| 2026-03-07 | DB interface is embedding + JSON insert, JSON returned on search | Match product behavior expectation | Data model and API prioritize JSON documents |
| 2026-03-07 | Default first database profile uses embedding dimension 512 | Establish concrete initial profile | Benchmarks and samples target 512-dimensional vectors |
| 2026-03-07 | Initial hard p95 target set to 20 ms | Set practical latency objective for v1 milestone | Performance gates and tuning use 20 ms as pass/fail line |
| 2026-03-07 | v1 filter requirement is JSON equality (`x = y`) with vector search | Enable targeted retrieval inside semantic search | Query path includes mandatory equality filter support |
| 2026-03-07 | Integration priority is desktop first, mobile second | Match immediate product usage | Roadmap phases prioritize desktop adapter and UX first |

### Scope Changes
When changing scope, document:
- Change requested:
- Reason:
- Effort impact:
- Timeline impact:
- Risk impact:
- Approved by:

## Sign-Off Checklist
- [ ] Final deliverables are still aligned with this file.
- [ ] Out-of-scope items are explicitly deferred.
- [ ] Acceptance criteria are still measurable.
- [ ] Decisions and scope changes are up to date.
