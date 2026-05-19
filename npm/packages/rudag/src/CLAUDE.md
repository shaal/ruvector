# rudag/src

TypeScript source for `@ruvector/rudag`. Compiled to `dist/` by `tsc` (both default and `tsconfig.esm.json`).

## Files

- `index.ts` — barrel re-exports of `RuDag`, `DagOperator`, `AttentionMechanism`, `DagNode`, `DagEdge`, `CriticalPath`, `RuDagOptions`, `DagStorage`, `MemoryStorage`, `createStorage`, `isIndexedDBAvailable`, `StoredDag`, `DagStorageOptions`. Includes a JSDoc quickstart.
- `dag.ts` — high-level `RuDag` TypeScript wrapper over the WASM `WasmDag` (node/edge add, topo sort, critical path, attention scores). Caches results to avoid WASM round-trips and validates inputs.
- `storage.ts` — `StoredDag` shape, IndexedDB-backed persistence with single-transaction atomic ops, plus an in-memory fallback and an `isIndexedDBAvailable` helper.
- `browser.ts` — re-exports `index` and adds `createBrowserDag(name)` that initializes a DAG backed by IndexedDB.
- `node.ts` — re-exports `index` and adds Node.js filesystem helpers with strict ID validation to prevent path traversal.
- `index.test.ts` — `node --test` test suite.

Each `.ts` source has compiled `.js`, `.d.ts`, and `.map` siblings.
