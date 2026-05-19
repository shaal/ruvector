# edge / benches

Criterion benchmarks for the edge crate.

## Important files
- `zkproof_bench.rs` - benchmarks the Bulletproofs-based zero-knowledge proof pipeline in `../src/plaid/zkproofs.rs` and `../src/plaid/zkproofs_prod.rs`.

## Run
- `cargo bench -p ruvector-edge` (HTML reports in `target/criterion/`).

## Related
- ZK source: `../src/plaid/`. Tuning notes: `../docs/zk_optimization_*.md`, `../docs/zk_performance_*.md`.
