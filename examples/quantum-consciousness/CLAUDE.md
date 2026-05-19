# quantum-consciousness

Research demo applying IIT 4.0 Phi to quantum-circuit measurement statistics (TPMs), exploring the relationship between entanglement and integrated information. Experimental / research-tier.

## Important files
- `Cargo.toml` — single bin `quantum-consciousness`, depends on `ruvector-consciousness`.
- `RESEARCH.md` — research writeup / methodology.
- `src/main.rs` — CLI entry; flags `--output <path>` (default `quantum_report.svg`) and `--depth <n>` (default 5).
- `src/data.rs` — generates quantum circuit TPMs of varying qubit counts.
- `src/analysis.rs` — Phi / emergence / collapse computation via `ruvector-consciousness`.
- `src/report.rs` — SVG/text report.

## Run
- `cargo run --release --bin quantum-consciousness -- --output report.svg --depth 5`.

## Tech stack
- `../../crates/ruvector-consciousness` (features: `phi`, `emergence`, `collapse`).
- `rand`, `rand_chacha`.

## Related
- Sibling: `../climate-consciousness` (same harness shape over climate teleconnections).
