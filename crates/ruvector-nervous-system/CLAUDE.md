# ruvector-nervous-system

Biologically-inspired nervous-system components for RuVector: dendritic coincidence detection with NMDA-like nonlinearity, hyperdimensional computing (HDC) for neural-symbolic AI, cognitive routing, Hopfield associative memory, spiking networks, BTSP / EWC / e-prop plasticity, and pattern-separation circuits.

## Important files

- `Cargo.toml` — deps: `rand`, `rand_distr`, `ndarray`, `parking_lot`, `thiserror`, `serde`, optional `rayon` (default `parallel`) and `bincode` (`serde` feature). Dev: `criterion`, `proptest`, `approx`.
- `src/lib.rs` — crate doc with worked examples (Dendrite, HDC, routing); module declarations.
- `src/lib_dendrite_only.rs` — dendrite-only build (alternative crate entry used by some downstream consumers).
- `HDC_IMPLEMENTATION.md`, `HOPFIELD.md` — top-level design docs.
- `docs/EWC_IMPLEMENTATION.md`, `docs/compete-implementation.md` — additional design docs.

## Module map (src/)

- `dendrite/` — `Dendrite`, `DendriticTree`, NMDA-like coincidence, plateau potentials, compartmental tree.
- `hdc/` — hyperdimensional computing: `vector.rs`, `ops.rs` (bind/bundle/permute), `similarity.rs`, `memory.rs`.
- `hopfield/` — Hopfield associative memory: `network.rs`, `capacity.rs`, `retrieval.rs`, `tests.rs`.
- `plasticity/` — `btsp.rs` (Behavioral Time-Series Pattern), `eprop.rs` (e-prop), `consolidate.rs` (EWC consolidation).
- `compete/` — competition / winner-take-all: `kwta.rs`, `wta.rs`, `inhibition.rs`.
- `separate/` — pattern separation: `dentate.rs`, `projection.rs`, `sparsification.rs`.
- `routing/` — cognitive routing: `circadian.rs`, `coherence.rs`, `predictive.rs`, `workspace.rs` (Global Workspace Theory).
- `eventbus/` — sharded event bus with backpressure: `event.rs`, `queue.rs`, `shard.rs`, `backpressure.rs` (+ `IMPLEMENTATION.md`).
- `integration/` — `postgres.rs`, `ruvector.rs`, `versioning.rs` external-system integrations.

## Tests & benches

- `tests/` — integration tests (`btsp_integration`, `eprop_tests`, `ewc_tests`, `integration`, `memory_bounds`, `retrieval_quality`, `throughput`, `workspace_integration`) plus a nested `tests/integration/` dir.
- `benches/` — `btsp_bench`, `eprop_bench`, `ewc_bench`, `hdc_bench`, `latency_benchmarks`, `pattern_separation`.
- `examples/` — `hopfield_demo.rs`, `workspace_demo.rs`, plus `examples/tiers/` (tier-1..tier-4 worked examples).

## Features

`default = ["parallel"]`; `serde` (enables bincode); `parallel` (enables rayon, used by EWC). Some throughput assertions are feature-gated for CI sanity.

## Related

- `crates/ruvector-cognitive-container` (witness chains in plasticity).
- `crates/ruvector-consciousness` (consumes HDC / workspace signals).
