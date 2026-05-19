# ruvllm-wasm / src

TypeScript source for `@ruvector/ruvllm-wasm`.

## Files
- `index.ts` - Public API. Exposes the `RuvLLMWasm` class with
  `create({ useWebGPU })`, `loadModel(url, { onProgress })`, and
  `generate(prompt, { maxTokens, temperature })`. Loads the
  underlying wasm-bindgen module from the Rust `ruvllm-wasm` crate.
- `types.ts` - Shared TypeScript types for model configuration,
  generate options, and result objects.

Each `.ts` ships with adjacent `.js`/`.d.ts` source maps.
