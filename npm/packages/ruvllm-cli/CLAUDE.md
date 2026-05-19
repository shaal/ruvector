# @ruvector/ruvllm-cli

Standalone CLI for LLM inference, benchmarking, and model management — runs local LLMs with Metal/CUDA acceleration. Installs as the `ruvllm` binary.

## Key files

- `package.json` — `@ruvector/ruvllm-cli` v0.1.0; main `dist/index.js`; bin `ruvllm` -> `./bin/ruvllm.js`.
- `tsconfig.json`.

## Subdirectories

- `bin/` — `ruvllm.js` launcher (`./bin/ruvllm.js`).
- `src/` — TypeScript source: `index.ts` (entry), `types.ts` (shared types) with checked-in compiled `.js`/`.d.ts`.

## Published API

- `.` -> `dist/index.js` (ESM+CJS via single entry).

## Scripts

- `build` -> `tsc`
- `prepublishOnly` -> `build`
- `test` -> `node --test test/*.test.js` (no `test/` present in this checkout)
- `typecheck` -> `tsc --noEmit`
- `clean` -> remove `dist`

## Related

- Rust crate: `crates/ruvllm-cli` (per `homepage`).
- Sibling: `npm/packages/ruvllm/` — heavier full-runtime + CLI variant (the `ruvllm` bin name is shared).
