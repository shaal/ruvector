# benchmarks/src/bin

Each file is a standalone `cargo` binary target declared in `../../Cargo.toml`. Most benchmarks load synthetic or RVF-encoded workloads and emit timing + accuracy reports to stdout.

## Important files
- `temporal_benchmark.rs` — temporal reasoning over time-stamped vectors.
- `vector_benchmark.rs` — raw vector op throughput (similarity, batch search).
- `swarm_regret.rs` — multi-agent regret minimization harness.
- `timepuzzle_runner.rs` — time-puzzle reasoning suite.
- `intelligence_assessment.rs` / `rvf_intelligence_bench.rs` — broad-spectrum intelligence test batteries (latter operates on RVF native format).
- `superintelligence.rs` — extended/stress variant.
- `agi_proof_harness.rs` — type-theoretic proof harness using `lean-agentic`.
- `acceptance_rvf.rs` — RVF acceptance suite.
- `wasm_solver_bench.rs` — WASM sublinear-solver throughput.

## Run
- `cargo run --release --bin <name>` from the crate root.
