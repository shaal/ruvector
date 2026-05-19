# @ruvector/graph-node

Native Node.js bindings (NAPI-RS) for the RuVector Graph Database — hypergraph support, Cypher queries, persistence. ~10x faster than the WASM equivalent.

## Key files

- `package.json` — `@ruvector/graph-node` v2.0.4; main `index.js`, types `index.d.ts`.
- `index.js` / `index.d.ts` — NAPI loader + TS types (delegates to platform `.node` binaries from optional deps).
- `test.js` — smoke test (`npm test`).
- `benchmark.js` — perf benchmark (`npm run benchmark`).
- `scripts/publish-platforms.js` — publishes per-platform sub-packages.

## Subdirectories

- `npm/` — per-platform sub-packages with their own `package.json` (darwin-arm64, darwin-x64, linux-arm64-gnu, win32-x64-msvc; linux-x64-gnu is also in the optional deps).

## Published API

Exports the Graph DB binding (constructors for graphs / Cypher executor). Platform binaries come via optional deps `@ruvector/graph-node-{linux,darwin,win32}-...` (all v2.0.4).

## Scripts

- `build:napi` -> `napi build --platform --release --cargo-cwd ../../../crates/ruvector-graph-node`
- `test` -> `node test.js`
- `benchmark` -> `node benchmark.js`
- `publish:platforms` -> `node scripts/publish-platforms.js`

## Related

- Rust crate: `crates/ruvector-graph-node` (referenced from `build:napi`).
- Sibling: `npm/packages/graph-data-generator/` (data producer for this DB).
