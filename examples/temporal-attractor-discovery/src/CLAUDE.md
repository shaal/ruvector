# temporal-attractor-discovery / src

Single-file source for the temporal-attractor-discovery demo binary.

## Important files
- `main.rs` - models a multi-regime time series (pulsar / FRB / X-ray binary style), builds a temporal coherence graph, and uses `ruvector_mincut` + `ruvector_coherence::spectral::estimate_fiedler` to detect each regime transition.

## Run
- `cargo run -p temporal-attractor-discovery --release`.

## Related
- Same pattern, different domain: `../../brain-boundary-discovery/src/`, `../../weather-boundary-discovery/src/`, `../../music-boundary-discovery/src/`.
