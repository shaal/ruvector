# earthquake-boundary-discovery

Research-tier example that uses `ruvector-mincut` (exact mincut) and `ruvector-coherence` (spectral) to discover tectonic-plate-style boundaries from earthquake event data. One of several "boundary-discovery" demos in the repo.

## Files

- `Cargo.toml` - Manifest; depends on `ruvector-mincut` (exact) and `ruvector-coherence` (spectral), plus `rand`.
- `src/main.rs` - The full example (~18KB).

## How to run

```bash
cargo run -p earthquake-boundary-discovery --release
```

## Tech stack

- Rust 2021. Workspace lints relaxed (research code).
- Internal crates: `ruvector-mincut`, `ruvector-coherence`.

## Related sibling examples

- `examples/brain-boundary-discovery`, `examples/cmb-boundary-discovery`, `examples/frb-boundary-discovery`, `examples/boundary-discovery` - other domain variants of the same algorithm pattern.
- `examples/gw-consciousness`, `examples/seizure-clinical-report` - similar research-tier coherence demos.
