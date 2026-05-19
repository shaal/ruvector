# pkg/

wasm-bindgen output for the RVF solver Rust crate (`crates/rvf/rvf-solver-wasm`).

- `rvf_solver.js` — wasm-bindgen JS glue.
- `rvf_solver.d.ts` — TypeScript declarations.
- `rvf_solver_bg.wasm` — compiled WASM (~135 KB), produced by `npm run build:wasm` and shrunk with `wasm-opt -Oz`.
- `.gitkeep`, `.npmignore`.

Pure build output; consumed by `../src/solver.ts`.
