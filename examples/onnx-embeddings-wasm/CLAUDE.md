# onnx-embeddings-wasm

`ruvector-onnx-embeddings-wasm` (standalone crate): browser/edge-runtime embedding pipeline using `tract-onnx` (compiles to WASM) and the `unstable_wasm` feature of HuggingFace `tokenizers`. Targets browsers, Cloudflare Workers, Deno, and other JS runtimes.

## Files

- `Cargo.toml` - Standalone package; `cdylib`+`rlib`; `console_error_panic_hook` default; release uses `opt-level="s"`, full LTO, wasm-opt `-Os`.
- `Cargo.lock` - Pinned lockfile.
- `src/` - Rust embedder, model loader, pooling, tokenizer.
- `loader.js` - JS-side WASM loader/glue.
- `parallel-embedder.mjs`, `parallel-worker.mjs` - Worker-pool embedding from JS.
- `test.mjs`, `test-full.mjs`, `test-parallel.mjs` - Node-based smoke tests.

## How to build

```bash
cd /home/user/ruvector/examples/onnx-embeddings-wasm
wasm-pack build --target web --release
node test.mjs
node test-parallel.mjs
```

## Tech stack

- Rust 2021 + `tract-onnx`, `tract-core`, `tokenizers`, `wasm-bindgen`, `web-sys`, `js-sys`.
- JS side: ES modules, Web Workers.

## Related

- Native sibling: `examples/onnx-embeddings` (uses `ort` runtime).
- WASM in-browser viewer: `examples/vwm-viewer`.
- Other WASM crates: `examples/wasm/ios`, `examples/prime-radiant/wasm`.
