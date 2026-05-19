# exo-core

Foundational crate of the EXO-AI substrate: defines core traits
(`Substrate`, `Witness`, `CoherenceRouter`, etc.) and the IIT-based
consciousness measurement + Landauer thermodynamics primitives. Every
other `exo-*` crate depends on this one.

## Files

- `Cargo.toml` — depends on `ruvector-core`, `ruvector-graph`, serde,
  thiserror, dashmap, uuid.
- `src/lib.rs` — public surface and re-exports.
- `src/traits.rs` — core trait set.
- `src/types.rs` — shared types (state IDs, witness records, etc.).
- `src/consciousness.rs` — Phi / IIT measurement primitives.
- `src/thermodynamics.rs` — Landauer-bound accounting.
- `src/substrate.rs` — `Substrate` trait + helpers.
- `src/witness.rs` — observer / witness pattern.
- `src/learner.rs` — base learning interface.
- `src/genomic.rs` — genome / configuration encoding.
- `src/coherence_router.rs` — routes coherent state between backends.
- `src/plasticity_engine.rs` — synaptic-style plasticity rules.
- `src/error.rs` — error enum.
- `src/backends/` — optional/feature-gated backends (neuromorphic +
  quantum stub).
- `tests/core_traits_test.rs` — trait contract tests.

## Build / Test

```bash
cargo build -p exo-core
cargo test  -p exo-core
```

## Related

- `../exo-backend-classical/` — primary consumer
- `../exo-hypergraph/`, `../exo-manifold/`, `../exo-temporal/`,
  `../exo-federation/`, `../exo-exotic/` — all depend on these traits.
