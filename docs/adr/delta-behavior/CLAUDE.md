# docs/adr/delta-behavior/

ADR-DB subseries for the **Delta Behavior** subsystem - the incremental, behavior-encoded delta representation used for propagating changes through ruvector indices and across nodes. Carved out of the main series after ADR-016 introduced the DDD architecture.

## ADRs

- `ADR-DB-001-delta-behavior-core-architecture.md` - core architecture and domain model.
- `ADR-DB-002-delta-encoding-format.md` - on-disk/wire encoding format.
- `ADR-DB-003-delta-propagation-protocol.md` - how deltas propagate between nodes.
- `ADR-DB-004-delta-conflict-resolution.md` - conflict resolution strategy.
- `ADR-DB-005-delta-index-updates.md` - how deltas update HNSW/IVF indices.
- `ADR-DB-006-delta-compression-strategy.md` - compression approach.
- `ADR-DB-007-delta-temporal-windows.md` - temporal windowing semantics.
- `ADR-DB-008-delta-wasm-integration.md` - WASM integration surface.
- `ADR-DB-009-delta-observability.md` - metrics, logs, traces.
- `ADR-DB-010-delta-security-model.md` - access control and integrity.

## Related

- `../ADR-016-delta-behavior-ddd-architecture.md` - parent ADR.
- `../../research/cognitive-frontier/delta-behavior-computational-paradigm.md` - exploratory background.
