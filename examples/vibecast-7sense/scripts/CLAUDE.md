# vibecast-7sense/scripts

Operational scripts for the 7sense workspace.

## Files
- `Cargo.toml` - Small standalone package providing the `performance_report` binary.
- `performance_report.rs` - Aggregates Criterion JSON output across crates into a human-readable performance report.
- `run_benchmarks.sh` - Runs the workspace benchmarks (`cargo bench`) and feeds the results into `performance_report.rs`.

## Run
```
./scripts/run_benchmarks.sh
```

## Related
- Benchmark targets: `../benches/`, `../crates/sevensense-benches/`, per-crate `benches/` directories.
