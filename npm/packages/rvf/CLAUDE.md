# @ruvector/rvf

Unified TypeScript SDK for the **RuVector Format (RVF)** — the binary vector storage format. Wraps either the native Node.js backend (`@ruvector/rvf-node`) or the WASM browser backend (`@ruvector/rvf-wasm`) behind one ergonomic `RvfDatabase` class. Also re-exports the higher-level AGI components from `@ruvector/rvf-solver`.

## Important files

- `package.json` — `@ruvector/rvf` v0.2.0. Main `dist/index.js`, types `dist/index.d.ts`. Dep: `@ruvector/rvf-node`. Optional deps: `@ruvector/rvf-wasm`, `@ruvector/rvf-solver`. Scripts: `build` (tsc), `test` (jest), `bench` (`tsx bench/index.ts`), `typecheck`.
- `src/index.ts` — barrel exporting `RvfDatabase`, `RvfBackend`, `NodeBackend`, `WasmBackend`, `resolveBackend`, `RvfError`, `RvfErrorCode`, `RvfSolver` (re-export from `@ruvector/rvf-solver`), plus all public types from `./types` (`DistanceMetric`, `CompressionProfile`, `HardwareProfile`, `RvfOptions`, filter/query/search/ingest/delete/compaction/status/segment/kernel/ebpf/witness/index types, `BackendType`, `DerivationType`).
- `src/database.ts` — `RvfDatabase` class. Delegates I/O to a `RvfBackend`. Use static factories `create`, `open`, `openReadonly`.
- `src/backend.ts` — `RvfBackend` interface + `NodeBackend`/`WasmBackend` implementations + `resolveBackend` helper.
- `src/errors.ts` — `RvfError`, `RvfErrorCode`.
- `src/types.ts` — exported type definitions.
- `src/externals.d.ts` — module declarations for optional backends.
- `dist/` — built ESM output (also has its own short CLAUDE.md).
- `tests/test-id-mapping.js` — small test fixture.

## Related

- Rust source: `crates/rvf*` (rvf-runtime, rvf-store, rvf-solver).
- Sibling packages: `@ruvector/rvf-node`, `@ruvector/rvf-wasm`, `@ruvector/rvf-solver`, `npm/packages/rvf-mcp-server`.
