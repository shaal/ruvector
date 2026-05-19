# ruvLLM / benches

Criterion benchmarks for ruvLLM. Each file is registered as a `[[bench]]` in `../Cargo.toml` (`harness = false`).

## Important files
- `pipeline.rs` - end-to-end inference pipeline bench.
- `router.rs` - router benches (`../src/router.rs`).
- `memory.rs` - memory subsystem benches (`../src/memory.rs`).
- `attention.rs` - attention kernels (`../src/attention.rs`, `simd_inference.rs`).
- `sona_bench.rs` - SONA continual-learning benches (`../src/sona/`).

## Run
- `cargo bench -p ruvllm` (or pick one with `--bench pipeline` etc.). HTML reports in `target/criterion/`.

## Related
- Bench driver binary: `../src/bin/bench.rs` and `benchmark_suite.rs`.
