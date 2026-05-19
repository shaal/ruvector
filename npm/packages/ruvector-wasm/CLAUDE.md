# @ruvector/wasm

Unified meta-package for RuVector WASM modules. No real code of its own — it pulls in five WASM subpackages and re-exports them under stable subpath imports.

## Key files

- `package.json` — `@ruvector/wasm` v0.1.30; ESM; main `./index.js`, types `./index.d.ts`.

Note: `index.js`, `index.d.ts`, and `README.md` are listed in `files` but are not present in this checkout (added at publish/build time).

## Published API / exports

- `.` -> default meta-export
- `./learning` -> `@ruvector/learning-wasm`
- `./economy` -> `@ruvector/economy-wasm`
- `./exotic` -> `@ruvector/exotic-wasm`
- `./nervous-system` -> `@ruvector/nervous-system-wasm`
- `./attention` -> `@ruvector/attention-unified-wasm`

## Scripts

- `build` -> echo (meta-package, nothing to build)
- `test` -> `node --test`
- `typecheck` -> `tsc --noEmit`
- `prepublishOnly` -> `typecheck`

## Related

- Sibling: `npm/packages/ruvector-wasm-unified/` (a richer TS API surface across the same WASM modules — distinct package).
- All `*-wasm` siblings under `npm/packages/`.
