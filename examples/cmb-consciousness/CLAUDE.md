# cmb-consciousness

CMB (Cosmic Microwave Background) consciousness explorer. Searches the CMB power spectrum for signatures of integrated information using IIT 4.0 Phi, causal emergence, and MinCut analysis. Produces an SVG report.

## Important files
- `Cargo.toml` - binary crate `cmb-consciousness`. Depends on `ruvector-consciousness` (features: `phi`, `emergence`, `collapse`) plus `rand` / `rand_chacha`. Research-tier lint relaxations.
- `RESEARCH.md` - ~33 KB writeup of the theoretical background, methodology, and results.
- `src/` - the analysis pipeline (data fetch / TPM construction / analysis / cross-frequency coupling / HEALPix utilities / report rendering).

## Run
- `cargo run -p cmb-consciousness --release -- --bins 16 --null-samples 100 --alpha 1.0 --output cmb_report.svg`.

## Related
- Same `ruvector-consciousness` family: `../gene-consciousness/` (gene regulatory networks instead of cosmology).
- Boundary-discovery analogues that use spectral/mincut graph tools instead of IIT: `../brain-boundary-discovery/`, `../weather-boundary-discovery/`, `../temporal-attractor-discovery/`, `../music-boundary-discovery/`.
