# seti-exotic-signals/src

Source for the SETI exotic-signals boundary-discovery binary.

## Files
- `main.rs` - Constants: `CHANNELS=128`, `TIMESTEPS=100`, `WIN_T=20`, `WIN_STEP=5`, `NULL_PERMS=100`, `SEED=2025`. Injects six exotic signal types into a sub-threshold spectrogram and tests whether per-window inter-channel coherence (`estimate_fiedler`) + `MinCutBuilder` localizes them better than amplitude thresholding.

## Related
- Parent: `../CLAUDE.md`.
- Sibling boundary-discovery binaries.
