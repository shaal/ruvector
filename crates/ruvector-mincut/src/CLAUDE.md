# ruvector-mincut/src

Source for the dynamic min-cut library. Top-level files plus one subdirectory per subsystem.

## Top-level files

- `lib.rs` — crate documentation, public re-exports (`MinCutBuilder`, `DynamicMinCut`, monitoring API).
- `error.rs` — crate-wide error enum.
- `time_compat.rs` — time-source abstraction (so the crate also compiles to wasm where `std::time::Instant` is unavailable).

## Algorithm subsystems

- `algorithm/` — exact + approximate / replacement-based algorithms.
- `subpolynomial/` — the headline subpolynomial-time scheme (O(n^{o(1)}) amortized update).
- `canonical/` — canonical decomposition: `dynamic/`, `source_anchored/`, `tree_packing/`.
- `core/` — shared abstractions across algorithms.
- `instance/` — `bounded.rs`, `stub.rs`, `traits.rs`, `witness.rs` instance representations.

## Decomposition / graph data structures

- `graph/`, `tree/`, `jtree/`, `linkcut/`, `euler/`, `expander/`, `cluster/`, `connectivity/`, `fragment/`, `fragmentation/`, `compact/`, `sparsify/`.

## Local / paper algorithms

- `localkcut/` — deterministic local k-cut discovery (paper-faithful 4-color coding).

## Proofs / auditing

- `certificate/`, `witness/` — proof certificates and witness chains.

## Cognition layer

- `snn/` — spiking neural network engine (attractor, causal, cognitive_engine, morphogenetic, neuron, synapse, network, optimizer, strange_loop).

## Perf / integration

- `optimization/` — caching, parallel, SIMD distance, dspar, wasm-batch tuning, benchmark helpers.
- `parallel/`, `pool/` — runtime parallelism primitives.
- `monitoring/` — event-driven notifications and metrics.
- `integration/`, `wrapper/` — external integration façades.
- `wasm/` — `agentic.rs`, `canonical.rs`, `mod.rs`, `simd.rs` — wasm-specific bindings/builders (separate from `crates/ruvector-mincut-wasm`).
