# ruvector-nervous-system/src

Implementation of biological / bio-inspired nervous-system components.

## Top-level

- `lib.rs` — primary crate entry; module declarations and worked examples.
- `lib_dendrite_only.rs` — alternate dendrite-only crate entry (used when downstream consumers only need the dendrite subsystem).

## Subsystems

- `dendrite/` — coincidence detection (NMDA-like) + compartmental tree.
- `hdc/` — hyperdimensional computing.
- `hopfield/` — Hopfield associative memory.
- `plasticity/` — BTSP, e-prop, EWC consolidation.
- `compete/` — winner-take-all / kWTA / inhibition.
- `separate/` — pattern separation (dentate, projection, sparsification).
- `routing/` — cognitive routing (circadian, coherence, predictive, Global Workspace).
- `eventbus/` — sharded event bus with backpressure.
- `integration/` — external integration (Postgres, RuVector, versioning).
