Workspace-level Criterion benchmarks for ruvector's research components. Each file is registered as a `[[bench]]` target somewhere in the workspace and is invoked via `cargo bench --bench <name>`.

Files:
- `attention_latency.rs` - latency of attention mechanisms (ScaledDotProduct, MultiHead, Flash, Linear, Hyperbolic, etc.) at seq_len=100. Target: <100 microseconds per mechanism.
- `learning_performance.rs` - MicroLoRA, SONA, and adaptive learning throughput / parameter efficiency.
- `neuromorphic_benchmarks.rs` - HDC (Hyperdimensional Computing), BTSP plasticity, and spiking neural network components.
- `plaid_performance.rs` - Plaid ZK proof generation, transaction processing, feature extraction, and LSH hashing. Expected: ~8us per proof, ~1.5us per transaction.

Conventions:
- All benches use `criterion` with `BenchmarkId` and `Throughput`.
- Deterministic seeded random vectors via a local `random_vector(dim, seed)` helper.
- Run individually with `cargo bench --bench <name>` or via `scripts/benchmark/run_benchmarks.sh`.

Related: see `../benchmarks/` for the load-testing/TS suite, `../crates/ruvector-bench/` for the standalone bench crate, and `../crates/ruvllm/benches/` for LLM-specific microbenchmarks.
