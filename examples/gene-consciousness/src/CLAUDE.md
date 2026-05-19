# gene-consciousness / src

Source for the `gene-consciousness` binary, split into pipeline stages.

## Important files
- `main.rs` - CLI entry; wires the stages below.
- `data.rs` - synthesises / loads gene regulatory network data (normal + oncogenic).
- `analysis.rs` - runs the IIT Phi / emergence / collapse analysis from `ruvector_consciousness`.
- `report.rs` - renders the results (text/SVG).

## Run
- `cargo run -p gene-consciousness --release`.

## Related
- Sibling: `../../cmb-consciousness/src/` (same dependency, CMB instead of gene networks).
