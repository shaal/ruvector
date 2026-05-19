# real-eeg-multi-seizure/src

Source for the CHB-MIT chb01 multi-seizure boundary-detection binary.

## Files
- `main.rs` - Constants: `NCH=16`, `SR=256`, `WIN_S=10`, `HALF_WIN=300s` each side of onset, `NULL_N=50`. Builds a `NFEAT = 120 corr + 48 band + 16 dom_freq` feature vector per window across 23 standard scalp channels (`LABELS`), runs `estimate_fiedler` from `ruvector-coherence::spectral`, and aggregates cross-seizure statistics.

## Related
- Parent: `../CLAUDE.md`.
- Same methodology applied to other domains: `examples/*-boundary-discovery/`.
