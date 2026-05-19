# cmb-boundary-discovery/src

Source for the CMB Cold Spot boundary-discovery binary.

## Files
- `main.rs` - Generates a synthetic 50x50 CMB patch with a Cold Spot (radius 8, dip -150 uK) surrounded by a hot ring (radius 10, +60 uK) on a Gaussian random field with `KERNEL_SIGMA=3.0` pixels. Compares spectral metrics (Fiedler + mincut from `ruvector-coherence` / `ruvector-mincut`) of the boundary ring vs interior vs `N_CONTROLS=20` background patches.

## Constants
- `SIZE=50`, `COLD_RADIUS=8.0`, `RING_RADIUS=10.0`, `COLD_DIP=-150 uK`, `RING_BUMP=60 uK`.

## Related
- Parent: `../CLAUDE.md`.
- Sibling boundary-discovery binaries.
