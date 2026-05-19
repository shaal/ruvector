# ruvector-postgres/src/workers

Background Workers for RuVector Postgres v2 — engine coordination, index maintenance, GNN training, and integrity monitoring.

## Files

- `mod.rs` — Module entry + architecture doc.
- `lifecycle.rs` — Worker lifecycle (start/stop/restart).
- `engine.rs` — Engine-coordination worker (query routing, load balancing).
- `maintenance.rs` — Index maintenance (compaction, cleanup, stats).
- `gnn.rs` — GNN incremental training worker.
- `integrity.rs` — Integrity monitor worker (mincut recomputation).
- `queue.rs` — Shared work queue.
- `ipc.rs` — Inter-process communication primitives for workers.
