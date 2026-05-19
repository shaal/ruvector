# edge-net/benches

Rust benchmark harness for edge-net core paths (compute, scheduler, swarm, credits).

## Important files
- `benchmark_runner.rs` — Cargo-discovered benchmark binary.
- `run_benchmarks.sh` — shell helper that builds in release and runs the benches with consistent flags.

## Run
- `cargo bench --features bench` from `../`.
- Or: `./run_benchmarks.sh`.

## Related
- Results / methodology: `../docs/benchmarks/`.
