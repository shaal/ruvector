# seizure-clinical-report

Research-tier example: uses `ruvector-mincut` (exact) and `ruvector-coherence` (spectral) on EEG-like seizure data to produce a clinical-style report. Same pattern as the other boundary-discovery examples but specialised for medical/neuro data.

## Files

- `Cargo.toml` - Manifest; depends on `ruvector-mincut` and `ruvector-coherence`, plus `rand`. Lints relaxed (research tier).
- `src/main.rs` (~27 KB) - Single-file demo: synthesise/load EEG, run mincut + spectral coherence, emit clinical report.

## How to run

```bash
cargo run -p seizure-clinical-report --release
```

## Tech stack

- Rust 2021. Internal: `ruvector-mincut` (exact), `ruvector-coherence` (spectral); `rand`.

## Related

- Sibling boundary-discovery demos: `examples/brain-boundary-discovery`, `examples/earthquake-boundary-discovery`, `examples/cmb-boundary-discovery`, `examples/frb-boundary-discovery`.
- Other clinical/medical examples: `examples/rvf/examples/medical_graphcut.rs`, `examples/rvf/examples/medical_imaging.rs`.
