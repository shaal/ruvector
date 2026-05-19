# @ruvector/wasm-unified

Unified TypeScript API surface for RuVector WASM — attention, learning, nervous-system, economy, and exotic features. Provides ergonomic per-domain entry points built from a single TS codebase.

## Key files

- `package.json` — `@ruvector/wasm-unified` v1.0.0; built with `tsup` (CJS+ESM+dts).
- `tsconfig.json`.
- `src/` — TypeScript source (with checked-in `.js`/`.d.ts` build copies).

## Source layout (`src/`)

- `index.ts` — top-level barrel.
- `attention.ts`, `learning.ts`, `nervous.ts`, `economy.ts`, `exotic.ts` — per-domain TypeScript façades over the underlying WASM modules.
- `types.ts` — shared types.

## Published API / exports

- `.` -> main barrel (CJS+ESM+dts)
- `./attention`, `./learning`, `./nervous`, `./economy`, `./exotic` — per-domain subpath exports.

## Scripts

- `build` -> `tsup` (CJS+ESM+dts, clean)
- `build:watch` -> watch mode
- `test` / `test:watch` -> vitest
- `typecheck` -> `tsc --noEmit`
- `lint` -> eslint
- `prepublishOnly` -> build

## Related

- Sibling: `npm/packages/ruvector-wasm/` (a much simpler meta-package re-exporting the same WASM packages without a TS layer).
- Wraps the same underlying `@ruvector/*-wasm` packages, plus exotic/nervous/attention/learning/economy modules.
