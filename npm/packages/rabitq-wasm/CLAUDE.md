# @ruvector/rabitq-wasm

RaBitQ 1-bit quantized vector index compiled to WebAssembly — 32x embedding compression with high-recall reranking. Targets browsers, Cloudflare Workers, Deno, and Bun.

## Key files

- `package.scoped.json` — `@ruvector/rabitq-wasm` v0.1.0; ESM; main `ruvector_rabitq_wasm.js`. (Note: file is named `package.scoped.json`, not `package.json` — likely the scoped variant of a wasm-bindgen output.)
- `.gitignore` — ignores the actual WASM artifacts (the `.wasm`, generated `.js`, `.d.ts`).

## Published API

Output of `wasm-bindgen` for the `ruvector-rabitq-wasm` Rust crate. Published files (per `files` field):

- `ruvector_rabitq_wasm_bg.wasm`
- `ruvector_rabitq_wasm.js`
- `ruvector_rabitq_wasm.d.ts`
- `ruvector_rabitq_wasm_bg.wasm.d.ts`

These are not committed here — they are produced at build/publish time.

## Related

- Rust crate: `crates/ruvector-rabitq-wasm` (per `repository.directory`).
- Sibling: `npm/packages/rabitq*` and other WASM packages (`ruvector-wasm`, `rvf-wasm`, etc.).
