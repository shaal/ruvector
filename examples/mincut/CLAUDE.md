# mincut

Collection of "exotic" demos exercising `ruvector-mincut`: temporal attractors, strange loops, causal discovery, time crystals, morphogenetic networks, neural optimizers, spiking-neural integration, temporal hypergraphs, federated loops, and benchmarks. Standalone crate (`[workspace]` declared in its Cargo.toml).

## Important files
- `Cargo.toml` — declares 10 `[[example]]` targets pointing to subdirectories; depends on `ruvector-mincut` (`monitoring`, `approximate`, `exact`).
- One subdirectory per example, each containing a `main.rs` (the `temporal_attractors` subdir is itself a standalone Cargo project with its own `Cargo.toml`).

## Run
- Any example: `cargo run --release --example <name>` (e.g. `causal_discovery`, `strange_loop`, `time_crystal`, `morphogenetic`, `neural_optimizer`, `benchmarks`, `snn_integration`, `temporal_hypergraph`, `federated_loops`).
- The `temporal_attractors` example also has its own standalone bin: `cd temporal_attractors && cargo run --release`.

## Tech stack
- `../../crates/ruvector-mincut` only.

## Related
- Boundary-discovery demos under `../*-boundary-discovery/` use the same crate with the `exact` feature.
