# climate-consciousness/src

Source modules for the climate-consciousness binary.

## Important files
- `main.rs` — CLI entry; orchestrates data → analysis → report.
- `data.rs` — generates / loads climate index time series.
- `analysis.rs` — runs Phi / emergence / collapse calculations via `ruvector-consciousness`.
- `report.rs` — writes the SVG/text summary.

## Build
- From parent: `cargo run --release`.
