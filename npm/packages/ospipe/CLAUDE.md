# @ruvector/ospipe

OSpipe SDK — RuVector-enhanced personal AI memory system for Screenpipe pipes. Provides semantic/vector search over local screen capture data.

## Key files

- `package.json` — `@ruvector/ospipe` v0.1.2; ESM (`"type": "module"`); main `dist/index.js`.
- `tsconfig.json` — TS config.
- `src/` — TypeScript source (also contains compiled `.js`/`.d.ts`).

## Source layout (`src/`)

- `index.ts` — main entry: ospipe SDK API.
- `wasm.ts` — WASM-backed implementation; published under the `./wasm` subpath export.

## Published API / exports

- `.` -> `dist/index.js` (main SDK)
- `./wasm` -> `dist/wasm.js`

## Scripts

- `build` -> `tsc`
- `prepublishOnly` -> `build`

## Peer deps

- `@screenpipe/js` (optional) — the pipe runtime.

## Related

- Repository origin: `examples/OSpipe/` (per `repository.directory`).
- Uses RuVector vector primitives under the hood (likely via `@ruvector/core` or wasm modules).
