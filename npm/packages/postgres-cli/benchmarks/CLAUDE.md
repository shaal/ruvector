# postgres-cli/benchmarks

SQL benchmark scripts for the RuVector PostgreSQL extension, runnable via `psql`. Included in the published npm tarball (`files: ["dist", "benchmarks", "README.md"]`).

## Files

- `ruvector_benchmark_optimized.sql` — benchmark SQL exercising RuVector functions (vector ops, attention, GNN, graph).
- `run_benchmarks_optimized.sql` — driver script that runs the benchmark with optimized session settings.
