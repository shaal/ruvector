# npm/core/native/linux-x64

Locally-built Linux x64 NAPI-RS native binding for `@ruvector/core`. Loaded by `../../src/index.ts` (and the CJS variant) before falling back to the published `@ruvector/core-linux-x64-gnu` platform package.

## Files

- `ruvector.node` - Compiled NAPI-RS dynamic library (~5 MB). Exports `VectorDb`, `CollectionManager`, `JsDistanceMetric`, `version`, `hello`, `getMetrics`, `getHealth`.
- `index.cjs` - CJS shim that requires `./ruvector.node`, then renames `VectorDb` to `VectorDB` and wraps async methods (`insert`, `insertBatch`, `search`, `delete`, `get`, `len`, `isEmpty`). Adds a `static withDimensions(dimensions)` factory that constructs a VectorDB with default options (`Cosine` metric, `./ruvector.db` storage).

## Related

- `../../platforms/linux-x64-gnu/` - The publishable form of the same artifact.
- `../../../../crates/ruvector-core` - Underlying Rust crate (the NAPI-RS binding crate compiles against this).
