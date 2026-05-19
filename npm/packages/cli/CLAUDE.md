# @ruvector/cli

Command-line interface for the RuVector vector database with self-learning hooks. Installs as the `ruvector` binary.

## Key files

- `package.json` — `@ruvector/cli` v0.1.28; main `dist/cli.js`; bin `ruvector`.
- `tsconfig.json` — TS config.
- `src/` — source TypeScript (also contains prebuilt JS/d.ts).

## Source layout (`src/`)

- `cli.ts` / `cli.js` — main CLI entry; built to `dist/cli.js` and invoked via the `ruvector` bin.
- `storage.ts` / `storage.js` — storage backend abstraction (e.g. Postgres via `pg`).

## Published API

The package is primarily a CLI; the compiled `cli.js` is also importable but main consumers use the `ruvector` binary.

## Scripts

- `build` -> `tsc`
- `clean` -> remove `dist` and tsbuildinfo
- `typecheck` -> `tsc --noEmit`
- `lint` -> ESLint on `src/*.ts`

## Deps

- `commander ^12` (runtime).
- `pg ^8` as optional dep (storage backend).

## Related

- Sibling `npm/packages/ruvector/` — separate, larger CLI/SDK.
- Likely thin wrapper around shared storage primitives; no direct Rust crate reference in `package.json`.
