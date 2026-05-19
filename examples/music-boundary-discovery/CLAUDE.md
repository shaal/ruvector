# music-boundary-discovery

Boundary-first genre discovery. Generates 300 synthetic songs across 5 genres with overlap zones, builds a k-NN similarity graph, and uses spectral bisection (Fiedler / mincut) to show that "Ambient Electronic" is a *boundary genre* - the last cluster to separate, the one that sits between worlds.

## Important files
- `Cargo.toml` - small binary crate; depends on `ruvector-mincut` (`exact`) + `ruvector-coherence` (`spectral`) + `rand 0.8`. Research-tier lint relaxations.
- `src/main.rs` - the whole demo: synthetic song generation, k-NN graph, spectral bisection, narrative output.

## Run
- `cargo run -p music-boundary-discovery --release`.

## Related
- Sibling boundary-discovery demos sharing the same template: `../brain-boundary-discovery/`, `../weather-boundary-discovery/`, `../temporal-attractor-discovery/`.
- IIT-flavoured analogues: `../cmb-consciousness/`, `../gene-consciousness/`.
