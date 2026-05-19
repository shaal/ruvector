# ruvllm-wasm

WASM bindings for RuvLLM — browser-compatible LLM inference runtime with WebGPU acceleration. Provides KV-cache management, memory pooling, chat templates (Llama3, Mistral, Qwen, Phi, Gemma), HNSW Router (150x faster), MicroLoRA (<1ms adaptation), SONA learning loops, and TypeScript-friendly types.

## Layout

- `Cargo.toml` — `crate-type = ["cdylib", "rlib"]`. wasm-bindgen / wasm-bindgen-futures / js-sys / web-sys stack. Default feature `console_error_panic_hook`; opt-in `webgpu` feature pulls a large set of `web-sys/Gpu*` features. `wasm-opt = false` (wasm-pack handles it).
- `INTEGRATION_SUMMARY.md` — top-level integration writeup.
- `src/lib.rs` — public `RuvLLMWasm`, `GenerateConfig`, `ChatMessageWasm`, `ChatTemplateWasm`, `KvCacheWasm`/`KvCacheConfigWasm`, `IntelligentLLMWasm` / `IntelligentConfigWasm`. JS quick-start lives in the doc comment.
- `src/bindings.rs` — additional wasm-bindgen surfaces.
- `src/hnsw_router.rs` — 150x-faster HNSW-backed router for prompt routing / retrieval.
- `src/micro_lora.rs` — sub-millisecond MicroLoRA adaptation.
- `src/sona_instant.rs` — SONA learning loop integration.
- `src/pi_quant_wasm.rs` — Pi-quant quantization helpers exposed to JS.
- `src/quant_bench_wasm.rs` — quant bench exposed to JS.
- `src/utils.rs` — small JS-interop utilities.
- `src/webgpu/` — WebGPU compute / buffers / shaders (feature `webgpu`).
- `src/workers/` — Web Worker pool and shared messaging.

## Tests / docs / examples

- `tests/web.rs` — wasm-bindgen-test driven.
- `tests/intelligent_wasm_test.rs` — `IntelligentLLMWasm` end-to-end.
- `docs/MICRO_LORA.md` — MicroLoRA design / usage.
- `examples/micro_lora_example.ts` — TypeScript usage demo.

## Related crates

- `crates/ruvllm` (workspace) — native RuvLLM runtime being wrapped.
- `crates/ruvector-hyperbolic-hnsw-wasm` — for hierarchy-aware retrieval.
