# rvlite

Lightweight, embeddable vector database with SQL, SPARQL, and Cypher
query surfaces. Bundles a WASM build of the Rust `rvlite` crate plus
a TS SDK + CLI that runs in Node, the browser, and edge runtimes.

## Important files
- `package.json` - npm metadata (`rvlite` v0.2.4, ESM `type: module`).
  Dual entry (`dist/index.mjs` / `dist/index.js`), subpath export
  `./wasm`, CLI binary `bin/cli.js`.
- `bin/cli.js` - Commander-based CLI (`rvlite ...`) using chalk, ora,
  and readline. Backed by `src/cli-rvf.ts` for RVF flows.
- `src/index.ts` - SDK entry. Exposes the `RvLite` class with
  `insert(vector, meta)`, `search(query, k)`, plus `sql()`, `cypher()`,
  and `sparql()` query helpers.
- `src/cli-rvf.ts` - RVF (Rust Virtual Function) integration used by
  the CLI.
- `.rvlite/db.json` - Default on-disk database file used during local
  development.
- `tsconfig.json` - TS configuration.

## Exports / scripts
- Main `dist/index.js` (CJS via esbuild) and `dist/index.mjs` (ESM).
  WASM subpath: `./wasm` -> `dist/wasm/rvlite.js`.
- `build` - `build:wasm` (wasm-pack on `../../../crates/rvlite`) +
  `build:sdk` (tsc + dual esbuild bundles).
- `test` - `node --test test/*.test.js`. `prepublishOnly` -> build.

## Key deps
- Runtime: `commander`, `chalk`, `ora`.
- Optional / peer: `@anthropic-ai/sdk`, `@ruvector/rvf-wasm`.

## Related
- Rust crate: `../../../crates/rvlite`.
- Optional companion: `@ruvector/rvf-wasm` (sibling npm package).
