# @ruvector/router

Semantic router for AI agents: vector-based intent matching with HNSW
indexing and SIMD acceleration. Native Node addon (napi-rs) wrapping
the Rust `ruvector-router-ffi` crate.

## Important files
- `package.json` - npm metadata (`@ruvector/router` v0.1.30). Declares
  `optionalDependencies` for each platform binary
  (`@ruvector/router-{linux,darwin,win32}-*`).
- `index.js` - Platform shim. Maps `process.platform`/`arch` -> per-
  platform package + `.node` filename and loads it.
- `index.d.ts` - TypeScript API: `DistanceMetric` enum, `DbOptions`,
  router/insert/search signatures.
- `test.js` - Smoke test (`npm test`).

## Exports / entry
- `main` -> `index.js`, `types` -> `index.d.ts`. Published files:
  `index.js`, `index.d.ts`, `README.md`.

## Scripts
- `build:napi` - `napi build --platform --release --cargo-cwd
  ../../../crates/ruvector-router-ffi`.
- `test` - `node test.js`.
- `publish:platforms` - publishes per-platform binary packages.

## Related
- Rust FFI crate: `../../../crates/ruvector-router-ffi`.
- Rust core: `../../../crates/ruvector-router-core`.
- CLI crate: `../../../crates/ruvector-router-cli`.
- Sibling: `../router-linux-arm64-gnu` (and other platform packages).
