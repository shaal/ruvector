# delta-behavior / src / bin

Cargo binaries shipped alongside the delta-behavior library.

## Important files
- `run_benchmarks.rs` - bench-runner binary, registered in `../../Cargo.toml` as `[[bin]] name = "run_benchmarks"`, gated by the `benchmarks` feature.

## Run
- `cargo run -p delta-behavior --release --bin run_benchmarks --features benchmarks`.

## Related
- Criterion benches: `../../benches/coherence_benchmarks.rs`.
