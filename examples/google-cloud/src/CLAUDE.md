# google-cloud/src

Source for the `gpu-benchmark` binary: a Clap-driven CLI plus an Axum HTTP server for Cloud Run.

## Files
- `main.rs` - Clap CLI entrypoint; subcommands include `Quick` (single-config bench with `--dims`, `--num-vectors`, `--num-queries`) and additional modes (full, server). Wires together the other modules.
- `benchmark.rs` - Core benchmark harness (vector / HNSW / distance workloads, hdrhistogram timings).
- `cuda.rs` - CUDA / GPU-path benchmark (simulated path in `cuda_sim.json` runs).
- `simd.rs` - CPU SIMD benchmarks.
- `self_learning.rs` - Self-learning model benchmarks combining RuVector's GNN + attention crates.
- `server.rs` - Axum server for running benchmarks remotely (Cloud Run entrypoint).
- `report.rs` - Report rendering / aggregation over saved JSON outputs.

## Related
- Outputs land in `../benchmark_results/`.
- Deployment scripts in `../deploy.sh` and `../cloudrun.yaml`.
