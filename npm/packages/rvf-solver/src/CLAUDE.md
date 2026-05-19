# src/

TypeScript source for `@ruvector/rvf-solver`. Compiled to `../dist/` via `tsc`.

- `index.ts` — package entry; re-exports `solver` API and types.
- `solver.ts` — thin wrapper around the `pkg/rvf_solver.js` WASM (Thompson Sampling, PolicyKernel, ReasoningBank).
- `types.ts` — shared TS types for solver inputs/outputs.
