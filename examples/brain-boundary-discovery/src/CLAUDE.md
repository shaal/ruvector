# brain-boundary-discovery / src

Single-file source for the demo binary.

## Important files
- `main.rs` - end-to-end demo. Generates 16-channel synthetic EEG with deterministic phase transitions, computes per-window correlation features, builds a coherence graph, and uses `ruvector_mincut::MinCutBuilder` + `ruvector_coherence::spectral::estimate_fiedler` to detect the pre-ictal boundary minutes before amplitude thresholds trigger.

## Run
- `cargo run -p brain-boundary-discovery --release`.

## Related
- Sibling sources for the same pattern: `../../weather-boundary-discovery/src/main.rs`, `../../music-boundary-discovery/src/main.rs`, `../../temporal-attractor-discovery/src/main.rs`.
