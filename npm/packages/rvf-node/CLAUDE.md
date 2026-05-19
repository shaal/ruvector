# @ruvector/rvf-node

Node.js native bindings for RuVector Format (RVF) — a persistent vector store implementation. Ships the `.node` binaries for all major platforms directly in this package (no per-platform sub-packages used at install time, though they're declared as optional deps too).

## Key files

- `package.json` — `@ruvector/rvf-node` v0.1.7; main `index.js`, types `index.d.ts`.
- `index.js` — NAPI loader.
- `index.d.ts` — TypeScript declarations.
- `rvf-node.darwin-arm64.node`, `rvf-node.darwin-x64.node`, `rvf-node.linux-arm64-gnu.node`, `rvf-node.linux-x64-gnu.node`, `rvf-node.win32-x64-msvc.node` — checked-in native binaries (~1 MB each).

## Published API

The RVF storage / vector API exposed by the underlying Rust crate via NAPI.

## Scripts

- `build` -> `napi build --platform --release`

## Optional deps (alt distribution path)

`@ruvector/rvf-node-{linux-x64-gnu, linux-arm64-gnu, darwin-x64, darwin-arm64, win32-x64-msvc}` all v0.1.7.

## Related

- Rust crate: under `crates/rvf/` (cf. sibling `rvf-solver` / `rvf-wasm`).
- Consumed by `ruvector` as the persistent-store backend (`@ruvector/rvf`).
