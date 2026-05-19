Shell wrappers around `cargo bench` for the workspace's Criterion benchmarks.

Files:
- `run_benchmarks.sh` - comprehensive benchmark runner across the workspace.
- `run_llm_benchmarks.sh` - runs all Criterion benchmarks for the `ruvllm` crate. Tuned for Mac M4 Pro hardware (per the script header).

Both call into the Rust benches under `../../benches/` and `../../crates/*/benches/`. Results land wherever Criterion writes them (typically `target/criterion/`).
