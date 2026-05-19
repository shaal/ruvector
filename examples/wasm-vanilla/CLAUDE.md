# wasm-vanilla

Single-file vanilla-JS demo of the ruvector WASM module — no build step, no bundler. Useful as a minimal integration template.

## Important files
- `index.html` — full self-contained page with inline CSS and inline JS that imports the WASM bundle.

## Run
- Serve over HTTP: `python -m http.server` then open `http://localhost:8000/`.
- Adjust the import path inside `index.html` to point at your `pkg/` output (e.g. from `../edge-net/pkg/` or any `wasm-pack` build).

## Tech stack
- Vanilla HTML/CSS/JS only.

## Related
- Sibling browser/WASM demos: `../pwa-loader` (PWA + RVF decoder), `../wasm-react` (React variant), `../wasm` (broader WASM playground), `../edge-net/dashboard` (production-grade UI).
