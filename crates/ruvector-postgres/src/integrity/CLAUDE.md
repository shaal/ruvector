# ruvector-postgres/src/integrity

Integrity control plane — Stoer-Wagner mincut gating for vector-search integrity. Holds integrity contracts (`IntegrityContract` with `min_recall`, `max_latency_ms`, etc.) and emits events as health transitions.

## Files

- `mod.rs` — `IntegrityContract` and the integrity service entrypoint.
- `mincut.rs` — Stoer-Wagner mincut implementation.
- `contracted_graph.rs` — Contracted graph data structure used during mincut.
- `events.rs` — Event types emitted on integrity state changes.
- `gating.rs` — Gating decisions based on mincut results.

## Pointers

- Consumed by `../healing/detector.rs` (uses integrity events to trigger remediation).
- Pure-Rust algorithm sources: `ruvector-mincut` (and `ruQu` for the quantum side).
