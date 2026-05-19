# npm/wasm

Source for the `@ruvector/wasm` npm package: WebAssembly bindings to the Rust vector database, with separate entry points for browser and Node.js. Provides a pure-JS fallback when the native NAPI-RS binding (`@ruvector/core`) isn't available, and is the only option in browser environments.

## Important files

- `package.json` - `@ruvector/wasm` v0.1.1. Conditional exports route to `dist/browser.*` in browser bundlers and `dist/node.*` in Node, with a generic `dist/index.*` default. Also exposes named subpaths `./browser` and `./node`.
- `tsconfig.json` - CommonJS build (CJS target ES2020, lib includes `DOM`, declarations on).
- `tsconfig.esm.json` - ES2020 module build extending the base config; declarations disabled (CJS build generates them).
- `src/` - TypeScript sources and pre-built `.js` / `.d.ts` artifacts (see `src/CLAUDE.md`).
- `.npmignore`, `LICENSE` (MIT).

## Exports

- `VectorDB` class - Unified async API: `init()`, `insert()`, `insertBatch()`, `search()`, `delete()`, `get()`, `len()`, `isEmpty()`, `getDimensions()`, `save()`, static `load()`.
- `detectSIMD()`, `version()`, `benchmark()` - Async utilities.
- Types: `VectorEntry`, `SearchResult`, `DbOptions`.
- Browser entry adds IndexedDB persistence (`saveToIndexedDB`, `loadFromIndexedDB`); Node entry has placeholder filesystem persistence.

## Scripts

- `npm run build:wasm` - Calls `wasm-pack build` against `../../crates/ruvector-wasm` twice, once with `--target bundler` (outputs to `pkg/`) and once with `--target nodejs` (outputs to `pkg-node/`).
- `npm run build:ts` - Runs both TypeScript configs (CJS + ESM) producing `dist/`.
- `npm run build` - Runs WASM then TS builds.
- `npm test` - `node --test dist/index.test.js`.
- `prepublishOnly` - Full build before publishing.

## Distributed contents

`dist/`, `pkg/` (bundler-target WASM artifacts), `pkg-node/` (Node-target WASM artifacts), `README.md`, `LICENSE`.

## Related

- `../../crates/ruvector-wasm` - Rust source compiled via `wasm-pack`; built artifacts land in `pkg/` and `pkg-node/`.
- `../core` - NAPI-RS native variant of the same engine (preferred when running on a supported Node.js platform).
- `../packages/ruvector/` - Meta-package that auto-selects between native and WASM backends.
