# @ruvector/core

High-performance vector database with HNSW indexing exposed to Node.js via
napi-rs. Wraps the Rust crate `ruvector-node` to deliver 50k+ inserts/sec
for AI/ML similarity and semantic search.

## Important files
- `package.json` - npm metadata (`@ruvector/core` v0.1.31). Declares
  `optionalDependencies` for each platform binary
  (`ruvector-core-{linux,darwin,win32}-*`).
- `index.js` - Platform shim. Detects `process.platform`/`arch`, then
  `require()`s the matching native `ruvector-core-<triple>` package.
  Throws a helpful error if the binary is missing.
- `index.d.ts` - TypeScript definitions describing the napi-exposed
  vector DB API.
- `test.js` - Smoke test executed by `npm test`.
- `tsconfig.json` - TS config for declaration-only consumption.
- `scripts/publish-platforms.js` - Helper to publish per-platform packages.

## Exports / entry
- `main` -> `index.js`, `types` -> `index.d.ts`.
- Published files: `index.js`, `index.d.ts`, `README.md` only.

## Scripts
- `build:napi` - `napi build --platform --release --cargo-cwd
  ../../../crates/ruvector-node`.
- `test` - `node test.js`.
- `publish:platforms` - Runs `scripts/publish-platforms.js`.

## Related
- Rust crate: `../../../crates/ruvector-node` (napi-rs source).
- Sibling platform packages: `ruvector-core-{linux-x64-gnu,linux-arm64-gnu,
  darwin-x64,darwin-arm64,win32-x64-msvc}`.
