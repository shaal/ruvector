# npm/wasm/src

TypeScript sources for `@ruvector/wasm` along with their pre-built `.js`, `.d.ts`, and `.map` artifacts. Three entry points are emitted so that bundlers, browsers, and Node.js each pick the right runtime path; the package's `exports` field in `../package.json` routes consumers automatically.

## Files

- `index.ts` - Auto-detecting entry. Inspects `typeof window` and `process.versions.node` then dynamically imports either `../pkg/ruvector_wasm.js` (bundler/browser target) or `../pkg-node/ruvector_wasm.js`. Exports the unified `VectorDB` class plus `detectSIMD()`, `version()`, `benchmark()`, and types (`VectorEntry`, `SearchResult`, `DbOptions`).
- `browser.ts` - Browser-only entry. Imports the bundler-target WASM, awaits `wasmModule.default()` to initialise, and adds `saveToIndexedDB()` / static `loadFromIndexedDB()` for persistence.
- `node.ts` - Node.js-only entry. Imports the nodejs-target WASM (no init call needed). Has placeholder `saveToFile()` / `loadFromFile()` that currently warn that filesystem persistence is unimplemented.
- `index.test.ts` - Standalone test script (not `node:test`). Imports from `./node` and exercises insert/batch/search/get/delete/len/isEmpty plus utility functions; exits with non-zero on failure.
- `*.js`, `*.d.ts`, `*.map` - Compiled outputs / source maps committed alongside the TS sources.

## Notes

- All three entry files instantiate the underlying WASM `VectorDB` with hard-coded `'cosine'` metric and `useHnsw=true`, even when `DbOptions.metric` is supplied. Caller-provided `metric` is currently ignored.
- The wrapper requires `await db.init()` before any other method (throws otherwise).

## Related

- `../../../crates/ruvector-wasm` - Rust crate compiled by `wasm-pack` into `../pkg/` and `../pkg-node/`.
- `../tsconfig.json`, `../tsconfig.esm.json` - Compiler configs.
