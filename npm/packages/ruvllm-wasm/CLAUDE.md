# @ruvector/ruvllm-wasm

WebAssembly bindings for browser-based LLM inference. Lets you load
GGUF-format models and run them in the browser with optional WebGPU
acceleration.

## Important files
- `package.json` - npm metadata (`@ruvector/ruvllm-wasm` v0.1.0).
  Dual import/require export wired to `dist/index.js` /
  `dist/index.d.ts`.
- `src/index.ts` - Public surface: example shows
  `RuvLLMWasm.create({ useWebGPU: true })`, model loading with
  progress, and `generate()` with `maxTokens` / `temperature`.
- `src/types.ts` - Shared types (model config, generate options,
  results).
- `tsconfig.json` - TS compile to `dist/`.

## Exports / entry
- `main` -> `dist/index.js`, `types` -> `dist/index.d.ts`. Published
  files: `dist`, `README.md`.

## Scripts
- `build` - `tsc`.
- `test` - `node --test test/*.test.js`.
- `typecheck`, `clean`, `prepublishOnly` (-> build).

## Key dev deps
- `@webgpu/types`, `typescript`, `@types/node`.

## Related
- Rust crate: `../../../crates/ruvllm-wasm` (referenced by
  `repository.directory`).
- Native counterpart: `@ruvector/ruvllm` + per-platform npm packages
  in sibling directories.
