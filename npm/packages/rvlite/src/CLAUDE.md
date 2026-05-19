# rvlite / src

TypeScript source for the `rvlite` SDK and CLI helpers.

## Files
- `index.ts` - SDK entry. Exports the `RvLite` class. Example shows
  `new RvLite({ dimensions: 384 })` then `insert`, `search`, `sql`,
  `cypher`, `sparql` calls. Built into `dist/index.js` (CJS) and
  `dist/index.mjs` (ESM) by `npm run build:sdk`.
- `cli-rvf.ts` - RVF (Rust Virtual Function) integration used by
  `bin/cli.js` to load the wasm-compiled rvlite engine and execute
  CLI subcommands.

The Rust source lives in `../../../crates/rvlite`; the wasm output is
copied into `dist/wasm/` by `npm run build:wasm`.
