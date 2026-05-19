# npm/tests/performance

Optional performance benchmarks. Skipped by the aggregate test driver unless `--perf` is passed because they take meaningfully longer than the unit/integration suites.

## Files

- `benchmarks.test.js` - Uses `node:test` to measure insert throughput (single and batch), search latency, and resource usage against `ruvector.VectorIndex` at realistic dimensions (e.g. 384). Includes helpers for formatting throughput / duration.

## Running

- `node --test performance/benchmarks.test.js`
- `node run-all-tests.js --perf` - Runs unit + integration + performance.
- `node run-all-tests.js --only=performance`

## Related

- `../unit/`, `../integration/` - Correctness tests; run before benchmarking.
- `../../../crates/ruvector-bench/` - Rust-side benchmarks against the same engine.
