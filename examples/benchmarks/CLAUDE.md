# benchmarks

Comprehensive benchmark suite for ruvector temporal reasoning, vector ops, swarm regret, RVF intelligence assessment, AGI proof harnesses, and WASM solver performance. Provides 10 CLI binaries that exercise core, RVF native format, and type-theory verified reasoning paths.

## Important files
- `Cargo.toml` — defines 10 `[[bin]]` targets (temporal-benchmark, vector-benchmark, swarm-regret, timepuzzle-runner, intelligence-assessment, rvf-intelligence-bench, superintelligence, agi-proof-harness, acceptance-rvf, wasm-solver-bench).
- `src/bin/` — one entry point per binary (see subdirectory).
- `tests/integration_tests.rs` — cross-binary integration coverage.

## Run / build
- Specific benchmark: `cargo run --release --bin temporal-benchmark` (replace name with any from the bin list above).
- With viz: `cargo run --release --features visualize --bin vector-benchmark`.
- Tests: `cargo test`.

## Tech stack
- `ruvector-core` (parallel feature), `rvf-types`/`rvf-crypto`/`rvf-wire` (native RVF format), `lean-agentic` (type-theoretic verified reasoning), `rayon`, `tokio`, `clap`, `indicatif`, `hdrhistogram`, `statistical`, `plotters` (optional).

## Related
- Native RVF format under `../../crates/rvf/`.
- Other intelligence demos: `../verified-applications`, `../refrag-pipeline`.
