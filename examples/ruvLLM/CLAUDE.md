# ruvLLM

Self-learning LLM example with LFM2-style inference, RuVector integration (storage / GNN / attention / graph), SONA continual-learning machinery (LoRA Ultra, EWC++, ReasoningBank, learning loops), optimized NEON/Metal kernels via `ruvllm` crate, and an embedded counterpart for ESP32 microcontrollers.

## Important files
- `Cargo.toml` - top-level crate. Features: `default = ["storage","metrics"]`, plus `server`, `real-inference` (candle/HF/tokenizers/memmap2), `hf-export`, `napi`, `parallel`, `candle`, `metal`, `inference-metal`, `full`. Many binaries (`ruvllm-demo`, `-server`, `-bench`, `-benchmark-suite`, `-simd-demo`, `-pretrain`, `-export`).
- `Cargo.lock`, `package.json` (N-API), `.gitignore`, `.cargo/` - build / packaging metadata.
- `task_specific_adapters.rs` - top-level reference file (LoRA adapter sketches).
- `src/` - library code (attention, inference, learning, memory, router, simd_inference, sona/, ...).
- `benches/`, `tests/`, `config/example.toml` - benchmarking, integration tests, sample config.
- `docs/` - SONA suite (LoRA Ultra, learning loops, EWC++, ReasoningBank, memory dreams), SPARC methodology docs, `index.md`.
- `esp32/`, `esp32-flash/` - companion embedded crates (own Cargo workspaces).
- `modules/plans/` - planning artifacts (RTF spec).

## Build / run
- `cargo run -p ruvllm --release --bin ruvllm-demo`.
- Server: `cargo run -p ruvllm --release --bin ruvllm-server --features server`.
- Benchmarks: `cargo bench -p ruvllm`.
- Real Candle inference with Metal: `--features inference-metal`.

## Related
- ESP32 deployment: `./esp32/`, `./esp32-flash/`. Memory + safety modules: `../OSpipe/`. Coherence safety: `../delta-behavior/`.
