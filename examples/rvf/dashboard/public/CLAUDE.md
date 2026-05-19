# rvf/dashboard/public

Vite `public/` directory: files served as-is from the dev server root and copied verbatim into `dist/` at build time.

## Files

- `rvf_solver_wasm.wasm` (~175 KB) - The RVF solver WASM binary; loaded at runtime by the dashboard.

## Related

- Source loader: `../src/solver.ts`.
- Built copy: `../dist/rvf_solver_wasm.wasm`.
