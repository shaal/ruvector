# exo-core/src

Library source defining the EXO-AI substrate's core trait surface.

## Files

- `lib.rs` — re-exports public traits / types.
- `traits.rs` — `Substrate`, `Witness`, `CoherenceRouter`, `Learner`,
  etc.
- `types.rs` — IDs, witness records, configuration types.
- `consciousness.rs` — IIT Phi primitives.
- `thermodynamics.rs` — Landauer-bound thermodynamic accounting.
- `substrate.rs` — substrate trait + helpers.
- `witness.rs` — observer pattern for cognitive state.
- `learner.rs` — base learning interface.
- `genomic.rs` — genome / config encoding.
- `coherence_router.rs` — coherent-state routing across backends.
- `plasticity_engine.rs` — plasticity rules.
- `error.rs` — `ExoError` enum (thiserror).

## Subdirectories

- `backends/` — feature-gated `neuromorphic` + `quantum_stub` backends
  with a `mod.rs` toggle.

## Related

- `../tests/core_traits_test.rs` — contract tests
- `../../exo-backend-classical/src/` — primary trait implementor
