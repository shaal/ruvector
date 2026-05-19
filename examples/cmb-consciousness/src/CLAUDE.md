# cmb-consciousness / src

Source for the `cmb-consciousness` binary. Each file is one stage of the IIT-on-CMB pipeline.

## Important files
- `main.rs` - CLI entry; parses `--bins`, `--null-samples`, `--alpha`, `--output`; orchestrates the stages below.
- `data.rs` - CMB power-spectrum acquisition and conversion into a Transition Probability Matrix (TPM).
- `analysis.rs` - Phi / emergence / collapse analysis built on `ruvector_consciousness`.
- `emergence_sweep.rs` - parameter sweep over emergence/collapse settings.
- `cross_freq.rs` - cross-frequency coupling analysis across multipole bins.
- `healpix.rs` - HEALPix helpers for the spherical CMB geometry.
- `report.rs` - SVG report rendering.

## Run
- `cargo run -p cmb-consciousness --release` (writes `cmb_report.svg` by default).

## Related
- Same dependency, different domain: `../../gene-consciousness/src/`.
