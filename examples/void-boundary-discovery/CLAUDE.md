# examples/void-boundary-discovery

Tests the "boundary-first" thesis on cosmic voids: void *walls / filaments* carry more structural information than either interiors or exteriors. Synthetic 2D cosmic web with 1000 galaxies and 7 voids, friends-of-friends graph with linking length 5.0.

## Files
- `Cargo.toml` - Binary crate `void-boundary-discovery`. Depends on `ruvector-mincut` (with `exact`), `ruvector-coherence` (with `spectral`), `rand`.
- `src/main.rs` - Generates galaxies + voids in a 100x100 box, builds a galaxy proximity graph, and compares Fiedler / mincut on boundary vs interior vs exterior subgraphs of each void.

## Run
```
cargo run --release -p void-boundary-discovery
```

## Tech stack
- Rust 2021, `ruvector-coherence`, `ruvector-mincut`, `rand`.

## Related
- Sibling boundary-discovery binaries.
