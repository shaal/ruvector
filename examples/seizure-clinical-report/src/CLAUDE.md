# seizure-clinical-report/src

Single-file source for the seizure clinical-report demo.

## Files

- `main.rs` (~27 KB) - Loads/synthesises EEG-style seizure data, runs `ruvector-mincut` (exact) for region separation and `ruvector-coherence` (spectral) for coherence scoring, and prints a clinical report.

## Related

- Parent: `examples/seizure-clinical-report/`.
- Crates: `crates/ruvector-mincut`, `crates/ruvector-coherence`.
