# seti-boundary-discovery

Research demo using MinCut + coherence to surface candidate boundaries in SETI-style signal data (separating noise from structured emissions). Part of the boundary-discovery family.

## Important files
- `Cargo.toml` — single-binary crate; depends on `ruvector-mincut` (`exact`) and `ruvector-coherence`.
- `src/main.rs` — synthetic signal graph + cut + coherence pipeline.

## Run
- `cargo run --release`.

## Tech stack
- `../../crates/ruvector-mincut`, `../../crates/ruvector-coherence`, `rand`.

## Related siblings
- Other boundary-discovery demos under `../*-boundary-discovery/`; richer SETI work in `../seti-exotic-signals`.
