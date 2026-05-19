# ecosystem-consciousness/src

Three-module Rust binary that runs the food-web IIT analysis pipeline.

## Files

- `main.rs` — CLI entry: parses `--output`, calls
  `data::generate_all_ecosystems()`, hands each to `analysis::`, and
  writes the aggregate `report::` SVG.
- `data.rs` — synthetic ecosystem generators (rainforest 12 species,
  monoculture 8, coral reef 10) returning energy-flow weighted graphs.
- `analysis.rs` — Phi computation, species-removal contribution
  ranking, and causal-emergence over trophic-level partitions; calls
  into `ruvector-consciousness` features `phi`, `emergence`, `collapse`.
- `report.rs` — SVG rendering of per-ecosystem Phi, top contributors,
  and emergence scores.

## Related

- `../RESEARCH.md` — theoretical background
- `../../../crates/ruvector-consciousness/` — IIT 4.0 library
