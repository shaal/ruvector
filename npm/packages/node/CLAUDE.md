# @ruvector/node

Unified Node.js entry point for the Ruvector ecosystem. Re-exports `@ruvector/core` (vector DB with HNSW + SIMD via N-API) and `@ruvector/gnn` (graph neural network primitives) under a single import so consumers can use the whole stack via `import { VectorDB, RuvectorLayer } from '@ruvector/node'`.

## Important files

- `package.json` — `@ruvector/node` v0.1.23. Main `dist/index.js`, ESM `dist/index.mjs`, types `dist/index.d.ts`. Subpath export `./gnn` (`dist/gnn.{js,mjs,d.ts}`). Deps: `@ruvector/core ^0.1.15`, `@ruvector/gnn ^0.1.15`. Scripts: `build` (ESM + CJS via separate tsc configs), `test` (`node --test`), `clean`.
- `tsconfig.json` / `tsconfig.cjs.json` / `tsconfig.esm.json` — split TS configs for dual ESM/CJS output.
- `src/` — TypeScript sources (`index.ts`, `gnn.ts`).

## Exports

From `index.ts`: `VectorDB`, `CollectionManager`, `version`, `hello`, `getMetrics`, `getHealth`, `DistanceMetric` plus types (`DbOptions`, `HnswConfig`, `QuantizationConfig`, `VectorEntry`, `SearchQuery`, `CoreSearchResult`, `CollectionConfig`, `CollectionStats`, `Alias`, `HealthResponse`, `Filter`) from `@ruvector/core`; `RuvectorLayer`, `TensorCompress`, `differentiableSearch`, `hierarchicalForward`, `getCompressionLevel`, `init as initGnn` plus `CompressionLevelConfig`, `GnnSearchResult` from `@ruvector/gnn`. Also a convenience default-export object.

From `gnn` subpath: just the GNN re-exports.

## Related

- Underlying Rust: `crates/ruvector` (core) and `crates/ruvector-gnn`.
- Sibling: `@ruvector/core`, `@ruvector/gnn` (other npm packages providing the actual bindings).
