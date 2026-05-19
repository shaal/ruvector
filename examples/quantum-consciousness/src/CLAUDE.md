# quantum-consciousness/src

Source modules for the quantum-consciousness binary.

## Important files
- `main.rs` — CLI entry; parses `--output` / `--depth` and orchestrates data → analysis → report.
- `data.rs` — generates quantum circuit TPMs.
- `analysis.rs` — Phi / emergence / collapse computations.
- `report.rs` — writes the SVG/text summary.

## Build
- From parent: `cargo run --release`.
