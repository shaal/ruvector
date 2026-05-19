# ruvllm-wasm/src

WASM-binding source root for RuvLLM.

## Top-level files

- `lib.rs` — crate doc + public WASM API: `RuvLLMWasm`, `GenerateConfig`, `ChatMessageWasm`, `ChatTemplateWasm`, `KvCacheWasm`, `IntelligentLLMWasm`, etc.
- `bindings.rs` — additional wasm-bindgen surfaces / glue.
- `hnsw_router.rs` — 150x faster HNSW-backed router for prompt routing / retrieval.
- `micro_lora.rs` — MicroLoRA (<1ms adaptation).
- `sona_instant.rs` — SONA learning loop integration.
- `pi_quant_wasm.rs` — Pi-quant quantization exposed to JS.
- `quant_bench_wasm.rs` — quant bench exposed to JS.
- `utils.rs` — small JS-interop helpers.

## Subdirectories

- `webgpu/` — WebGPU compute / buffer / shader plumbing (feature `webgpu`).
- `workers/` — Web Worker pool, messaging, feature detection.
