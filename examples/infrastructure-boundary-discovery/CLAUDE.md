# infrastructure-boundary-discovery

Research demo using MinCut + spectral coherence to discover boundaries in an infrastructure network (e.g. discovering bottleneck cuts in topologies). Part of a family of "boundary-discovery" demos.

## Important files
- `Cargo.toml` — single-binary crate; depends on `ruvector-mincut` (`exact`) and `ruvector-coherence` (`spectral`).
- `src/main.rs` — entry point that generates a synthetic infrastructure graph and runs cut + coherence analysis.

## Run
- `cargo run --release`.

## Tech stack
- `../../crates/ruvector-mincut`, `../../crates/ruvector-coherence`, `rand`.

## Related siblings
- Similar boundary-discovery demos: `../boundary-discovery`, `../brain-boundary-discovery`, `../cmb-boundary-discovery`, `../earthquake-boundary-discovery`, `../frb-boundary-discovery`, `../health-boundary-discovery`, `../market-boundary-discovery`, `../music-boundary-discovery`, `../pandemic-boundary-discovery`, `../seti-boundary-discovery`, `../void-boundary-discovery`, `../weather-boundary-discovery`.
