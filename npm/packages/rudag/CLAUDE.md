# @ruvector/rudag

Fast DAG (Directed Acyclic Graph) library backed by Rust/WASM. Provides topological sort, critical path, attention-based importance scoring, task scheduling, and dependency resolution, with IndexedDB auto-persistence in browsers and a filesystem option in Node. Includes a `rudag` CLI and dual browser/node entry points.

## Important files

- `package.json` — `@ruvector/rudag` v0.1.0. Main / module / types under `dist/`. Bin: `rudag` → `bin/cli.js`. Subpath exports: `.`, `./browser`, `./node`, `./wasm` (with split `pkg/` for bundler and `pkg-node/` for Node). Dep: `idb`. Build steps invoke `wasm-pack` on `../../../crates/ruvector-dag-wasm` and then `tsc`.
- `src/index.ts` — barrel exporting `RuDag`, `DagOperator`, `AttentionMechanism`, plus types (`DagNode`, `DagEdge`, `CriticalPath`, `RuDagOptions`) and storage exports (`DagStorage`, `MemoryStorage`, `createStorage`, `isIndexedDBAvailable`, `StoredDag`, `DagStorageOptions`).
- `src/dag.ts` — `RuDag` high-level wrapper around the WASM `WasmDag` (validates input, caches results, persists via storage).
- `src/storage.ts` — IndexedDB-backed `DagStorage`, in-memory fallback `MemoryStorage`, and `createStorage`/`isIndexedDBAvailable` helpers.
- `src/browser.ts` — browser-specific entry exposing `createBrowserDag(name)` with IndexedDB persistence.
- `src/node.ts` — Node-specific entry with filesystem storage and path-traversal guards.
- `src/index.test.ts` — test suite.
- `bin/cli.js` — `rudag` CLI (lazy-loads core to keep startup fast).
- `pkg/` — `wasm-pack` bundler target output (for browsers/bundlers).
- `pkg-node/` — `wasm-pack` nodejs target output (CommonJS).
- `tsconfig.json` / `tsconfig.esm.json` — split TS configs.

## Scripts

- `build:wasm` (bundler + node), `build:ts` (tsc + ESM tsc), `build`, `test`, `prepublishOnly`.

## Related

- Rust source: `crates/ruvector-dag-wasm` (and `crates/ruvector-dag`).
