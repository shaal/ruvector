# gw-consciousness

Research demo applying IIT (Integrated Information Theory) Phi computations from `ruvector-consciousness` to gravitational-wave background data. Generates analyses and a long-form report.

## Files

- `Cargo.toml` - Manifest; depends on `ruvector-consciousness` with features `phi`, `emergence`, `collapse`. Bin `gw-consciousness`.
- `RESEARCH.md` - Background and methodology notes.
- `src/main.rs` - Entry point wiring data -> analysis -> report.
- `src/data.rs` - GW data sourcing/synthesis.
- `src/analysis.rs` - IIT/coherence analysis routines.
- `src/report.rs` (~17KB) - Report generator.

## How to run

```bash
cargo run -p gw-consciousness --release
```

## Tech stack

- Rust 2021. Internal crate `ruvector-consciousness`; `rand`, `rand_chacha`.
- Lints relaxed (research tier).

## Related

- Sibling consciousness demos: `examples/brain-boundary-discovery`, `examples/cmb-consciousness`, `examples/ecosystem-consciousness`, `examples/gene-consciousness`, `examples/climate-consciousness`.
- Underlying crate: `crates/ruvector-consciousness`.
