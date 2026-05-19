# examples/real-eeg-multi-seizure

Boundary-first seizure detection on real EEG data: parses all seven documented seizure files from the CHB-MIT Scalp EEG Database (patient chb01) and runs boundary-first detection on each, then computes cross-seizure statistics (mean warning time, detection rate, Fiedler consistency, per-channel informativeness).

## Files
- `Cargo.toml` - Binary crate `real-eeg-multi-seizure`. Depends on `ruvector-coherence` (with `spectral`) and `rand`.
- `src/main.rs` - Parses CHB-MIT EDF files (16 channels x 256 Hz x 10-second windows), builds per-window coherence features (120 channel pairs + 48 band + 16 dominant frequencies), runs `estimate_fiedler` to localize the boundary, and aggregates statistics across 7 seizures.

## Run
```
cargo run --release -p real-eeg-multi-seizure
```
(Expects CHB-MIT chb01 EDF files in the working directory; see `src/main.rs` for the `LABELS` channel order.)

## Tech stack
- Rust 2021, `ruvector-coherence::spectral`, `rand`.

## Related
- Synthetic-signal counterpart: `examples/boundary-discovery/`.
- Other domain boundary-discovery demos: `../{cmb,frb,market,void,seti-exotic-signals}-boundary-discovery/`.
