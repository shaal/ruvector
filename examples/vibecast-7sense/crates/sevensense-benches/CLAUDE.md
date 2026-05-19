# sevensense-benches

Workspace benchmarking helper crate (publish = false) that depends on every other `sevensense-*` crate so benchmarks can exercise the full stack.

## Files
- `Cargo.toml` - Depends on every `sevensense-*` crate plus workspace utilities.
- `src/lib.rs`, `src/utils.rs` - Shared bench helpers.
- `benches/` - Criterion targets: `api_benchmark.rs`, `clustering_benchmark.rs`, `embedding_benchmark.rs`, `hnsw_benchmark.rs`.

## Run
```
cargo bench -p sevensense-benches
# or via the wrapper
../../scripts/run_benchmarks.sh
```

## Related
- Workspace-level benches: `../../benches/`.
- Reporting: `../../scripts/performance_report.rs`.
