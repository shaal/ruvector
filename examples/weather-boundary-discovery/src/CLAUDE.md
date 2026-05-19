# weather-boundary-discovery / src

Single-file source for the weather-boundary-discovery demo binary.

## Important files
- `main.rs` - generates synthetic multi-channel weather (temperature, pressure, humidity, variance), builds a temporal coherence graph, and uses `ruvector_mincut` + `ruvector_coherence::spectral::estimate_fiedler` to flag the regime change days before any single-variable threshold fires.

## Run
- `cargo run -p weather-boundary-discovery --release`.

## Related
- Same pattern, different domain: `../../brain-boundary-discovery/src/`, `../../temporal-attractor-discovery/src/`, `../../music-boundary-discovery/src/`.
