# ruvector

The flagship npm package — self-learning vector database for Node.js. Hybrid search, Graph RAG, FlashAttention-3, HNSW, 50+ attention mechanisms, ONNX semantic embeddings, MCP integration, and a self-installing decompiler/optimizer. Auto-selects native (Rust), RVF (persistent), or WASM stub backends at runtime.

## Key files

- `package.json` — `ruvector` v0.2.25; main `dist/index.js`; bin `ruvector` -> `bin/cli.js`.
- `tsconfig.json`, `.npmignore`, `LICENSE`.
- `HOOKS.md`, `PACKAGE_SUMMARY.md` — design/feature docs.
- `ruvector-0.1.1.tgz` — packed legacy tarball (kept in tree).

## Subdirectories

- `bin/` — published CLI + MCP server JS bundles (`cli.js`, `mcp-server.js`).
- `examples/` — `api-usage.js`, `cli-demo.sh`.
- `scripts/` — `verify-dist.js` (publish prep verification).
- `src/` — TS source: barrel + `core/`, `services/`, `workers/`, `analysis/`, `decompiler/`, `optimizer/`.
- `test/` — Node integration / benchmark scripts (`integration.js`, `cli-commands.js`, `benchmark-*.js`, etc.).
- `wasm/` — prebuilt decompiler WASM module (`ruvector_decompiler_wasm`).

## Published API

`src/index.ts` re-exports everything from `./types`, `./core`, `./services`, and exposes runtime helpers:

- `getImplementationType()`, `isNative()`, `isRvf()` for backend selection.
- VectorDb / search APIs delegated to `@ruvector/core` (native) or `@ruvector/rvf` (persistent), falling back to a stub.

## Scripts

- `build` -> `tsc` + copy ONNX wasm package.json
- `verify-dist` -> `node scripts/verify-dist.js`
- `prepack`, `prepublishOnly` -> build + verify
- `test` -> `node test/integration.js && node test/cli-commands.js`

## Key deps

- Runtime: `@ruvector/attention`, `@ruvector/core`, `@ruvector/gnn`, `@ruvector/sona`, `@modelcontextprotocol/sdk`, `chalk`, `commander`, `js-beautify`, `ora`.
- Optional: `@ruvector/rvf` (persistent store).
- Peer (optional): `@ruvector/diskann`, `@ruvector/pi-brain`, `@ruvector/router`, `@ruvector/ruvllm`.

## Related

- Rust crates: many — primarily `crates/ruvector-core` (consumed via `@ruvector/core`).
- Sibling: `npm/packages/cli/` (separate, lighter CLI).
