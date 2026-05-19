# rvf/dashboard

`rvf-causal-atlas-dashboard`: TypeScript + Vite + Three.js single-page dashboard that visualizes RVF causal atlases, light curves, planetary systems, Dyson spheres, coherence heatmaps, etc. Loads the `@ruvector/rvf-solver` WASM package from `../../../npm/packages/rvf-solver`.

## Files

- `package.json` - npm manifest; scripts `dev`, `build`, `preview`; deps `three`, `d3-*`, `@ruvector/rvf-solver`.
- `tsconfig.json`, `vite.config.ts` - TS + Vite config.
- `index.html` - HTML entry.
- `src/` - TypeScript source (api/solver/ws plus views, charts, components, three.js scenes).
- `public/rvf_solver_wasm.wasm` - Solver WASM served at dev time.
- `dist/` - Built static bundle (gitted) including the WASM and split chunks.

## How to run

```bash
cd /home/user/ruvector/examples/rvf/dashboard
npm install
npm run dev
# or production:
npm run build && npm run preview
```

## Tech stack

- TypeScript 5, Vite 6, Three.js 0.170, D3 (scale/axis/shape/selection).
- Consumes RVF solver via local `file:` dep.

## Related

- Parent crate: `examples/rvf/`.
- WASM package: `npm/packages/rvf-solver`.
- Server-side example: `examples/rvf/examples/causal_atlas_dashboard.rs`.
