# climate-consciousness

Research demo applying IIT Phi (Integrated Information Theory 4.0) to climate teleconnection systems (ENSO, NAO, etc.) to quantify "consciousness-like" integration across coupled subsystems. Experimental / research-tier.

## Important files
- `Cargo.toml` — single bin `climate-consciousness`, depends on `ruvector-consciousness`.
- `src/main.rs` — CLI entry point.
- `src/data.rs` — climate index dataset construction.
- `src/analysis.rs` — Phi / emergence / collapse computations.
- `src/report.rs` — SVG/text report generation.

## Run
- `cargo run --release --bin climate-consciousness` (accepts `--output` and similar CLI flags; check `main.rs`).

## Tech stack
- `../../crates/ruvector-consciousness` (features: `phi`, `emergence`, `collapse`).
- `rand`, `rand_chacha`.

## Related
- Sibling consciousness demos: `../quantum-consciousness` (same harness shape over quantum TPMs).
