# wasm-react

Browser demo wiring the `ruvector-wasm` build into a React UI.
Initializes a `WorkerPool` (one Web Worker per hardware thread),
persists vectors via `IndexedDBPersistence`, and exposes search +
benchmark controls. Working SPA — but requires the `ruvector-wasm`
package to be built first (it pulls files from `../../crates/ruvector-wasm`).

## Important files

- `package.json` — Vite + React 18 dev server (`npm run dev`),
  build (`npm run build`), preview (`npm run preview`). No Tailwind.
- `vite.config.js` — sets `Cross-Origin-{Opener,Embedder}-Policy` so
  WASM threads work; excludes `@ruvector/wasm` from Vite optimization;
  serves on port `3000`.
- `index.html` — minimal mount point for `main.jsx`.
- `main.jsx` — boots React and renders `<App />`.
- `App.jsx` (~13 KB) — initializes `WorkerPool` + `IndexedDBPersistence`
  from `../../crates/ruvector-wasm/src/{worker-pool,indexeddb}.js`,
  loads the WASM glue from
  `../../crates/ruvector-wasm/pkg/ruvector_wasm.js`, manages 384-d
  vectors with SIMD detection, search results, and a benchmark panel.

## Run

```bash
# 1. Build the sibling WASM crate first
wasm-pack build crates/ruvector-wasm --target web --release
# 2. Install + serve
cd examples/wasm-react
npm install
npm run dev    # -> http://localhost:3000
```

## Tech stack

- React 18, Vite 5
- `../../crates/ruvector-wasm/` (provides `worker-pool.js`,
  `indexeddb.js`, and the `pkg/` wasm-pack output)
- Browser APIs: Web Workers, IndexedDB, COOP/COEP for cross-origin
  isolation

## Related

- `../nodejs/` — same `ruvector` idea but Node CommonJS
- `../decompiler-dashboard/` — React+TS dashboard (no WASM)
- `../exo-ai-2025/crates/exo-wasm/examples/browser_demo.html` — simpler
  vanilla-HTML WASM demo
