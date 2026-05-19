# temporal-attractor-discovery

Temporal attractor boundary detection. Models astrophysical-style multi-regime time series (pulsar magnetospheric switching, FRB activity cycles, X-ray binary state changes) where dynamical regime shifts are *invisible* to amplitude detectors but obvious in the topology of a temporal coherence graph. Demonstrates discovering MULTIPLE hidden state transitions.

## Important files
- `Cargo.toml` - small binary crate. Depends on `ruvector-mincut` (`exact`) + `ruvector-coherence` (`spectral`) + `rand 0.8`. Research-tier lint relaxations.
- `src/main.rs` - the full demo: synthetic multi-regime time series, temporal coherence graph, spectral / mincut analysis to detect each regime change.

## Run
- `cargo run -p temporal-attractor-discovery --release`.

## Related
- Sibling boundary-discovery demos: `../brain-boundary-discovery/`, `../weather-boundary-discovery/`, `../music-boundary-discovery/`.
- IIT analogues: `../cmb-consciousness/`, `../gene-consciousness/`.
