# music-boundary-discovery / src

Single-file source for the music-boundary-discovery demo binary.

## Important files
- `main.rs` - synthesises 300 songs across 5 genres, builds a k-NN similarity graph, and uses `ruvector_mincut` + `ruvector_coherence::spectral::estimate_fiedler` to show that "Ambient Electronic" is a boundary genre.

## Run
- `cargo run -p music-boundary-discovery --release`.

## Related
- Same pattern, different domain: `../../brain-boundary-discovery/src/`, `../../weather-boundary-discovery/src/`, `../../temporal-attractor-discovery/src/`.
