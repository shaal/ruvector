# examples/frb-boundary-discovery

Boundary-first discovery on a synthetic CHIME-like Fast Radio Burst catalog. Generates ~200 FRBs modeled on the CHIME/FRB Catalog 1 (arXiv:2106.04352) with injected sub-populations, builds a multi-parameter similarity graph, runs spectral bisection + min-cut to find population boundaries, and compares against a simple dispersion-measure threshold.

## Files
- `Cargo.toml` - Binary crate `frb-boundary-discovery`. Depends on `ruvector-mincut` (with `exact`), `ruvector-coherence` (with `spectral`), `rand`.
- `src/main.rs` - Builds a k=7 nearest-neighbor graph (sigma=0.28) over FRB feature vectors and validates the recovered boundary with 100 null permutations; seeded with the arXiv ID (`2106_04352`).

## Run
```
cargo run --release -p frb-boundary-discovery
```

## Tech stack
- Rust 2021, `ruvector-coherence`, `ruvector-mincut`, `rand`.

## Related
- Sibling boundary-discovery binaries in `examples/*-boundary-discovery/`.
