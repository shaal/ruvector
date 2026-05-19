# exo-temporal

Temporal memory coordinator for the EXO-AI substrate: short-term and
long-term stores, causal links, quantum-style decay, anticipation, and
consolidation. Wraps timestamps with a causal structure so the substrate
can replay / consolidate experiences without violating ordering.

## Files

- `Cargo.toml` — depends on `exo-core`, `ruvector-domain-expansion`,
  dashmap, parking_lot, chrono.
- `src/lib.rs` — re-exports.
- `src/types.rs` — shared temporal types.
- `src/causal.rs` — causal-link structure.
- `src/short_term.rs`, `src/long_term.rs` — STM / LTM stores.
- `src/quantum_decay.rs` — decay model.
- `src/anticipation.rs` — predictive lookups.
- `src/consolidation.rs` — STM -> LTM consolidation.
- `src/transfer_timeline.rs` — timeline used by transfer protocols.
- `tests/temporal_memory_test.rs` — coverage.

## Build / Test

```bash
cargo build -p exo-temporal
cargo test  -p exo-temporal
```

## Related

- `../../benches/temporal_bench.rs`
- `../exo-federation/src/transfer_crdt.rs`
