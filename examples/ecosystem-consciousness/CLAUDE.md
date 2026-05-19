# ecosystem-consciousness

Research demo applying Integrated Information Theory (IIT) Phi to food
web networks as an ecosystem-resilience metric. Generates three
synthetic ecosystems (tropical rainforest, agricultural monoculture,
coral reef), computes Phi + species-removal contributions + causal
emergence, and emits an SVG report. Runnable, prints results and writes
the SVG to disk.

## Important files

- `Cargo.toml` — single `ecosystem-consciousness` bin; depends on
  `ruvector-consciousness` (`../../crates/ruvector-consciousness`) with
  features `phi`, `emergence`, `collapse`, plus `rand` + `rand_chacha`.
  Most of the file is workspace-wide lint relaxations (`allow` of
  pedantic/style lints).
- `RESEARCH.md` — background on IIT, the food-web analogy, expected
  phenomena (high-Phi rainforest, low-Phi monoculture), and limitations.
- `src/main.rs`, `src/data.rs`, `src/analysis.rs`, `src/report.rs` —
  CLI entry, ecosystem fixtures, Phi/emergence analysis, SVG rendering.

## Run

```bash
cargo run -p ecosystem-consciousness -- --output ecosystem_report.svg
```

## Tech stack

- Pure Rust, no async, no external numeric crates beyond `rand`
- Backed by `../../crates/ruvector-consciousness` (IIT 4.0 toolbox)

## Related

- `../exo-ai-2025/research/06-federated-collective-phi/` — distributed
  Phi (CRDT/consensus version of the same metric)
- `../exo-ai-2025/research/07-causal-emergence/` — emergence detection
  in cleaner form
