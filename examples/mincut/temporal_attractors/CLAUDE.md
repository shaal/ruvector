# mincut/temporal_attractors

Standalone Cargo project (its own `[workspace]`) demonstrating temporal attractor networks with MinCut convergence analysis. Also wired in as the `temporal_attractors` example from the parent `mincut` crate.

## Important files
- `Cargo.toml` — `temporal-attractors-mincut-demo` bin `temporal-attractors`; depends on `../../../crates/ruvector-mincut`.
- `Cargo.lock`.
- `src/main.rs` — entry point.

## Run
- Standalone: `cargo run --release` from this dir.
- Via parent: `cargo run --release --example temporal_attractors` from `../`.

## Related
- Sibling research: `../../temporal-attractor-discovery`.
