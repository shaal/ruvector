# brain-boundary-discovery

Pre-seizure detection from EEG correlation graph boundaries. Generates 600 s of synthetic 16-channel EEG (256 Hz) progressing through normal -> pre-ictal -> seizure -> post-ictal, then shows that graph-structural boundary detection catches the pre-ictal hypersynchronization ~45 s BEFORE simple amplitude thresholds fire.

## Important files
- `Cargo.toml` - small binary crate, depends on `ruvector-mincut` (with `exact`) and `ruvector-coherence` (with `spectral`), plus `rand 0.8`. Research-tier crate so most lints are downgraded.
- `src/main.rs` - the full demo: synthetic EEG generator, sliding-window feature extraction, mincut/Fiedler-based boundary discovery, comparison vs amplitude threshold.

## Run
- `cargo run -p brain-boundary-discovery --release`.

## Related
- Sibling boundary-discovery examples sharing the same pattern: `../weather-boundary-discovery/`, `../music-boundary-discovery/`, `../temporal-attractor-discovery/`.
- Same family of consciousness/IIT studies: `../cmb-consciousness/`, `../gene-consciousness/`.
- Real (not synthetic) EEG analysis using the same building blocks: `../real-eeg-analysis/`.
