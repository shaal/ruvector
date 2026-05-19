# @ruvector/ospipe-wasm

WASM bindings for OSpipe — a RuVector-enhanced personal AI memory system for browsers (think "Screenpipe-style" capture + vector search). This directory is a publish-only placeholder: the actual WASM bundle (`pkg/ospipe.js`, `pkg/ospipe_bg.wasm`, `pkg/ospipe.d.ts`) is produced by `wasm-pack` against the Rust source and copied here at publish time.

## Important files

- `package.json` — `@ruvector/ospipe-wasm` v0.1.0. Main `pkg/ospipe.js`, types `pkg/ospipe.d.ts`, ESM `module`. Files published: `pkg/ospipe_bg.wasm`, `pkg/ospipe.js`, `pkg/ospipe.d.ts`, `pkg/ospipe_bg.wasm.d.ts`. No scripts.

## Related

- Source/example app: `examples/OSpipe` (referenced by `repository.directory`).
- Sibling WASM placeholder packages: `npm/packages/acorn-wasm`, `npm/packages/ruvector-cnn` (which actually ships its WASM in-tree).
