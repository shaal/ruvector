# examples/cmb-boundary-discovery

Boundary-first discovery applied to the CMB Cold Spot: shows that the *ring* around the temperature depression is more spectrally anomalous than the interior. Synthetic data models the Cruz et al. 2008 profile (central -150 uK dip, +60 uK hot ring) on top of a spatially correlated Gaussian random field.

## Files
- `Cargo.toml` - Binary crate `cmb-boundary-discovery`. Depends on `ruvector-mincut` (with `exact`), `ruvector-coherence` (with `spectral`), `rand`.
- `src/main.rs` - Builds a 50x50 pixel map with the Cold Spot profile + spatial kernel of sigma 3 pixels, constructs a galaxy/pixel proximity graph, and compares Fiedler value + mincut on boundary vs interior vs control regions across 20 controls.

## Run
```
cargo run --release -p cmb-boundary-discovery
```

## Tech stack
- Rust 2021, `ruvector-coherence`, `ruvector-mincut`, `rand`.

## Related
- Sibling boundary-discovery binaries in `examples/*-boundary-discovery/` apply the same pattern to different domains.
