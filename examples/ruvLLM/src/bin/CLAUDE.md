# ruvLLM / src / bin

Binaries shipped with the ruvLLM crate. All are registered in `../../Cargo.toml`.

## Important files
- `demo.rs` - `ruvllm-demo` end-to-end interactive demo.
- `server.rs` - `ruvllm-server` HTTP server (Axum, requires `--features server`).
- `bench.rs` - `ruvllm-bench` quick benchmark driver.
- `benchmark_suite.rs` - `ruvllm-benchmark-suite` larger benchmark suite.
- `simd_demo.rs` - `ruvllm-simd-demo` showcase of SIMD inference paths (`../simd_inference.rs`).
- `pretrain.rs` - `ruvllm-pretrain` small pretraining script.
- `export.rs` - `ruvllm-export` HuggingFace exporter (requires `--features hf-export`).

## Run
- `cargo run -p ruvllm --release --bin <name>` (add features as required).

## Related
- Library glue: `../orchestrator.rs`, `../inference.rs`, `../router.rs`. SONA: `../sona/`. Sample config: `../../config/example.toml`.
