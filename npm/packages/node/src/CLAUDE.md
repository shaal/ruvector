# @ruvector/node/src

TypeScript sources for `@ruvector/node`. Compiled to `dist/` by two tsc configs (ESM and CJS) — emitted `.js`, `.mjs`, `.d.ts`, and source maps are present alongside the `.ts` sources here.

## Files

- `index.ts` — main entry. Re-exports `VectorDB`, `CollectionManager`, `version`, `hello`, `getMetrics`, `getHealth`, `DistanceMetric` and all related types from `@ruvector/core`; re-exports `RuvectorLayer`, `TensorCompress`, `differentiableSearch`, `hierarchicalForward`, `getCompressionLevel`, `init as initGnn` and types from `@ruvector/gnn`. Provides a convenience default export.
- `gnn.ts` — focused subpath export. Re-exports only the GNN APIs from `@ruvector/gnn` for `@ruvector/node/gnn`.
