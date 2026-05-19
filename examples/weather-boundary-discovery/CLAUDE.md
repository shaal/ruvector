# weather-boundary-discovery

Detecting hidden weather regime changes via boundary-first discovery. Temperature follows a smooth sinusoid - you cannot see regime shifts from temperature alone - but variance, pressure, humidity, and correlation structure change sharply at the boundary. A temporal coherence graph detects *when* the regime changed days before any thermometer threshold fires.

## Important files
- `Cargo.toml` - small binary crate. Depends on `ruvector-mincut` (`exact`) + `ruvector-coherence` (`spectral`) + `rand 0.8`. Research-tier lint relaxations.
- `src/main.rs` - the whole demo: synthetic multi-channel weather series, sliding-window correlation features, spectral / mincut detection.

## Run
- `cargo run -p weather-boundary-discovery --release`.

## Related
- Sibling boundary-discovery demos: `../brain-boundary-discovery/`, `../music-boundary-discovery/`, `../temporal-attractor-discovery/`.
- IIT-flavoured analogues: `../cmb-consciousness/`, `../gene-consciousness/`.
