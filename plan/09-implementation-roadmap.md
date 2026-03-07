# 09 - Implementation Roadmap

## Phase 0 - Bootstrap (Week 1)
Deliverables:
- workspace crate split
- config and error scaffolding
- CI baseline (`fmt`, `clippy`, `test`)

Exit criteria:
- empty engine opens/closes with valid manifest.

## Phase 1 - Durable Storage Core (Weeks 2-3)
Deliverables:
- WAL append/replay
- vector/JSON document segment writer and reader
- ID allocator and tombstones

Exit criteria:
- crash-restart integration tests pass.

## Phase 2 - Graph Index MVP (Weeks 4-6)
Deliverables:
- graph data structure
- insert/search/delete basic flow
- brute-force baseline path

Exit criteria:
- recall and latency baseline measurable via harness.

## Phase 3 - NGT-Style Improvements (Weeks 7-8)
Deliverables:
- insertion/search parameter tuning (`ef`, exploration)
- optional seed index (VP-tree)
- background graph repair worker

Exit criteria:
- recall@10 and p95 latency hit target envelope.

## Phase 4 - RAG Layer (Weeks 9-10)
Deliverables:
- metadata filtering
- hybrid retrieval hooks
- context assembly and scoring policy

Exit criteria:
- end-to-end RAG memory retrieval demo is stable.

## Phase 5 - API/CLI And Desktop Integration (Weeks 11-12)
Deliverables:
- library API stabilization
- CLI commands for admin and query flows
- desktop integration adapter and health/reporting flows

Exit criteria:
- all major flows executable via CLI and desktop integration.

## Phase 6 - Hardening And Mobile Follow-Up (Weeks 13-14)
Deliverables:
- snapshot/compaction
- corruption detection + repair tooling
- observability (metrics/logging/tracing)
- mobile integration follow-up plan and adapter spike

Exit criteria:
- long-run soak tests and recovery drills pass.

## Phase 7 - Beta Readiness (Weeks 15-16)
Deliverables:
- migration tests
- benchmark regression gates
- docs and runbooks complete

Exit criteria:
- beta tag published with reproducible benchmark report.

## Phase 8 - Post-v1 Backlog
- quantization plugin architecture (still separate from NGTQ codebase)
- mobile-first SDK ergonomics and offline sync research (still local-only core)
- tiered storage and cold index shards

## Tracking Format
For each phase maintain:
- milestone checklist
- risk log
- benchmark deltas
- unresolved design decisions
