# @ruvector/acorn-wasm

ACORN (predicate-agnostic filtered HNSW) compiled to WebAssembly. Provides high-recall vector search with arbitrary metadata filters for browsers, Cloudflare Workers, Deno, and Bun. This directory is currently a placeholder for the published npm package — the actual WASM artifacts (`ruvector_acorn_wasm.js`, `ruvector_acorn_wasm_bg.wasm`, `.d.ts`) are produced by the corresponding Rust crate's `wasm-pack` build and copied here at publish time.

## Important files

- `package.scoped.json` — the scoped package manifest (`@ruvector/acorn-wasm` v0.1.0). Lists `main: ruvector_acorn_wasm.js`, `types: ruvector_acorn_wasm.d.ts`, ESM `module`, and `files` to publish (WASM + JS glue + type declarations + README).
- `.gitignore` — ignores generated WASM artifacts so they're not committed.

## Exported API (when built)

WASM bindings exposing the ACORN filtered HNSW index: build index, insert vectors with metadata, query with k-NN plus arbitrary predicate filter callbacks.

## Build

Built externally via `wasm-pack build` against the Rust crate. No npm scripts defined here.

## Related

- Rust source: `crates/ruvector-acorn-wasm` (referenced by `repository.directory` in package.scoped.json).
- Sibling: `npm/packages/ruvector-cnn` (similar WASM artifact-only package pattern).
