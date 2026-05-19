# exo-temporal/src

## Files

- `lib.rs` — public re-exports.
- `types.rs` — `Timestamp`, `MemoryRecord`, related types.
- `causal.rs` — causal-link / lamport-style structure.
- `short_term.rs` — bounded STM store (dashmap-backed).
- `long_term.rs` — durable LTM store.
- `quantum_decay.rs` — decay model (probabilistic forgetting).
- `anticipation.rs` — predictive lookups over the timeline.
- `consolidation.rs` — STM -> LTM consolidation pipeline.
- `transfer_timeline.rs` — timeline view for transfer protocols.

## Related

- `../tests/temporal_memory_test.rs`
- `../../exo-federation/src/transfer_crdt.rs`
