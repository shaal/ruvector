# delta-behavior / benches

Criterion benchmarks for the coherence core. Gated behind the `benchmarks` feature in `../Cargo.toml`.

## Important files
- `coherence_benchmarks.rs` - micro-benches for coherence computation, transition checks, and attractor evaluation.

## Run
- `cargo bench -p delta-behavior --features benchmarks` (HTML reports in `target/criterion/`).

## Related
- Driver binary: `../src/bin/run_benchmarks.rs`.
- Library under test: `../src/lib.rs`, `../src/simd_utils.rs`.
