# src/core/

Heart of the `ruvector` TypeScript layer — safe wrappers around the native Rust modules, WASM bindings, and pure-TS engines. Each file ships TS + compiled JS/d.ts.

- `index.ts` — barrel for the core subsystem.

## Engines / wrappers

- `adaptive-embedder.ts`, `neural-embeddings.ts`, `onnx-embedder.ts`, `onnx-optimized.ts` — embedding backends (adaptive, neural, ONNX MiniLM, optimized ONNX).
- `intelligence-engine.ts`, `learning-engine.ts`, `parallel-intelligence.ts`, `parallel-workers.ts` — self-learning intelligence / parallel orchestration.
- `attention-fallbacks.ts` — pure-TS attention implementations as fallback when native is missing.
- `neural-perf.ts`, `tensor-compress.ts` — perf measurement + tensor compression utilities.
- `cluster-wrapper.ts`, `coverage-router.ts`, `diff-embeddings.ts`, `graph-algorithms.ts`, `ast-parser.ts` — supporting analysis helpers.

## Wrappers around sibling packages

- `agentdb-fast.ts`, `diskann-wrapper.ts`, `gnn-wrapper.ts`, `graph-wrapper.ts`, `router-wrapper.ts`, `rvf-wrapper.ts`, `sona-wrapper.ts` — thin TS adapters with type conversion for `@ruvector/*` modules.

## Subdirectories

- `onnx/` — ONNX embedding WASM loader (`loader.js`, `pkg/`).
