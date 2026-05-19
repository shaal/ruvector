# src/

TypeScript source for `ruvector`. Compiled to `../dist/` via `tsc`.

- `index.ts` — main entry; auto-selects native (`@ruvector/core`) vs RVF (`@ruvector/rvf`) vs stub backend at load time; exposes `getImplementationType()`, `isNative()`, `isRvf()` and re-exports `types`, `core/*`, `services/*`.
- `types.ts` — shared TypeScript type definitions.

## Subdirectories

- `core/` — safe wrappers / adapters around native + WASM engines (vector DB, attention fallbacks, intelligence engine, learning engine, neural embeddings, ONNX embedder, parallel workers, graph algorithms, GNN/router/RVF/SONA/diskann wrappers).
- `services/` — higher-level services (e.g. embedding service).
- `workers/` — Node worker_threads orchestration (native worker, benchmark, types).
- `analysis/` — static analysis modules (`complexity`, `patterns`, `security`).
- `decompiler/` — JS plain-files implementing the self-decompiler (model decompiler, splitter, reconstructor, etc.).
- `optimizer/` — JS modules driving the settings/context optimizer for hooks.
