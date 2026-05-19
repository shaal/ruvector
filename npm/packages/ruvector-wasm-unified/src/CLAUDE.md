# src/

TypeScript source for `@ruvector/wasm-unified`. Compiled with `tsup` into `../dist/` (CJS+ESM+dts).

- `index.ts` — main barrel; re-exports the per-domain façades.
- `attention.ts` — façade over `@ruvector/attention-unified-wasm`.
- `learning.ts` — façade over `@ruvector/learning-wasm`.
- `nervous.ts` — façade over `@ruvector/nervous-system-wasm`.
- `economy.ts` — façade over `@ruvector/economy-wasm`.
- `exotic.ts` — façade over `@ruvector/exotic-wasm`.
- `types.ts` — shared TS types used across façades.

Each `.ts` ships alongside its compiled `.js` and `.d.ts`.
