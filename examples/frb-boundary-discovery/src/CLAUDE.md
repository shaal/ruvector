# frb-boundary-discovery/src

Source for the FRB-population boundary-discovery binary.

## Files
- `main.rs` - Generates `N_FRB=200` Fast Radio Bursts modeled on the CHIME/FRB Catalog 1 distributions with injected sub-populations, builds a k=7 (`K_NN=7`, `SIGMA=0.28`) feature-similarity graph, runs spectral bisection + min-cut, and compares the structural boundary to a naive DM threshold. Null permutations: 100. Seed: `2106_04352` (arXiv ID).

## Related
- Parent: `../CLAUDE.md`.
- Sibling boundary-discovery binaries.
