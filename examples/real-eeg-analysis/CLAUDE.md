# real-eeg-analysis

Real-data counterpart to `../brain-boundary-discovery/`. Parses clinical EEG from the CHB-MIT Scalp EEG Database (PhysioNet), file `chb01_03.edf` (seizure at seconds 2996-3036), and runs multi-scale boundary-first detection in pure Rust (no Python deps). EDF binary parsing is done in-process.

## Important files
- `Cargo.toml` - binary crate. Depends on `ruvector-mincut` (`exact`) + `ruvector-coherence` (`spectral`) + `rand`. Research-tier lint relaxations.
- `src/main.rs` - end-to-end pipeline: EDF parser, multi-scale windows (5/10/30 s), artifact rejection, enhanced spectral features, running baseline normalisation, patient-specific null model, mincut/Fiedler boundary detection.
- `data/chb01-summary.txt` - human-readable summary of the CHB-MIT recording (sampling rate, 16 channel montage, seizure annotations) used for the demo.

## Run
- `cargo run -p real-eeg-analysis --release` (expects the `.edf` to be downloaded separately; see comments at the top of `src/main.rs`).

## Related
- Synthetic version with the same algorithm: `../brain-boundary-discovery/`.
- Same boundary-discovery family: `../weather-boundary-discovery/`, `../music-boundary-discovery/`, `../temporal-attractor-discovery/`.
