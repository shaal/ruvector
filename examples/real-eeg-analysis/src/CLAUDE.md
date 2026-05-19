# real-eeg-analysis / src

Single-file binary that runs boundary-first seizure detection on real CHB-MIT EEG data.

## Important files
- `main.rs` - reads `chb01_03.edf` directly (in-process EDF binary parser), extracts a 16-channel window around the seizure (sec 2696..3296), computes multi-scale features (5/10/30 s windows, 50 % overlap), applies a patient-specific null model, and detects the boundary via `ruvector_mincut::MinCutBuilder` + `ruvector_coherence::spectral::estimate_fiedler`.

## Run
- `cargo run -p real-eeg-analysis --release` (place the downloaded `chb01_03.edf` where `main.rs` expects it - see the constants at the top of the file).

## Related
- Synthetic version: `../../brain-boundary-discovery/src/main.rs`.
