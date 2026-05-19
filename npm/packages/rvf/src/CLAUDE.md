# rvf/src

TypeScript source for `@ruvector/rvf`. Compiled to `dist/` by `tsc`.

## Files

- `index.ts` — barrel exporting `RvfDatabase`, `RvfBackend`, `NodeBackend`, `WasmBackend`, `resolveBackend`, errors, types, and `RvfSolver` re-export.
- `database.ts` — `RvfDatabase` main class. Backend is private; use static factories `create`, `open`, `openReadonly`. Provides `ingestBatch`, `query`, `delete`, `compact`, `status`, etc.
- `backend.ts` — `RvfBackend` abstract interface and `NodeBackend`/`WasmBackend` implementations plus `resolveBackend(type)` helper.
- `errors.ts` — `RvfError` class and `RvfErrorCode` enum.
- `types.ts` — public TypeScript types (`DistanceMetric`, `CompressionProfile`, `HardwareProfile`, `RvfOptions`, `RvfFilterValue`, `RvfFilterExpr`, `RvfQueryOptions`, `RvfSearchResult`, `RvfIngestResult`, `RvfIngestEntry`, `RvfDeleteResult`, `RvfCompactionResult`, `CompactionState`, `RvfStatus`, `DerivationType`, `RvfKernelData`, `RvfEbpfData`, `RvfSegmentInfo`, `BackendType`, `RvfIndexStats`, `RvfWitnessResult`).
- `externals.d.ts` — ambient module declarations for optional native/WASM peer packages.

Each `.ts` source also has compiled `.js`, `.d.ts`, and `.map` siblings present in-tree.
