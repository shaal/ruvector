# @ruvector/rvf-solver

RVF self-learning temporal solver — Thompson Sampling, PolicyKernel, and ReasoningBank wrapped in a small TS façade over a WASM core.

## Key files

- `package.json` — `@ruvector/rvf-solver` v0.1.7; main `dist/index.js`, types `dist/index.d.ts`.
- `tsconfig.json`, `.npmignore`.

## Subdirectories

- `src/` — TypeScript source (`index.ts`, `solver.ts`, `types.ts`).
- `dist/` — TS build output (currently just `solver.js`).
- `pkg/` — wasm-bindgen output for the Rust solver (`rvf_solver.js/.d.ts/_bg.wasm`).
- `test/` — `solver.test.mjs` runnable test.

## Published API

- `.` -> `dist/index.js` (typed via `dist/index.d.ts`).

`solver.ts` wraps the WASM solver and exposes the Thompson Sampling / PolicyKernel / ReasoningBank surface.

## Scripts

- `build` -> `tsc`
- `build:wasm` -> `cargo build --release --target wasm32-unknown-unknown --manifest-path ../../crates/rvf/rvf-solver-wasm/Cargo.toml && wasm-opt -Oz ... -o pkg/rvf_solver_bg.wasm`

## Related

- Rust crate: `crates/rvf/rvf-solver-wasm/`.
- Sibling: `npm/packages/rvf-wasm/` (lower-level RVF wasm microkernel), `npm/packages/rvf-node/` (native node bindings).
