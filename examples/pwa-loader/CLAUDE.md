# pwa-loader

Progressive Web App that decodes RVF (`.rvf`) cognitive seeds and witness bundles entirely in-browser using a WASM module. Supports file drop, file picker, and QR-code scanning. Working demo, no build step.

## Important files
- `index.html` — UI markup (drop zone, scanner, results, theme toggle).
- `app.js` — main app; loads `rvf_wasm_bg.wasm` (override path via `window.RVF_WASM_PATH`), parses RVQS 64-byte headers in pure JS.
- `style.css` — dark/light theme styling.
- `sw.js` — service worker (offline-first cache).
- `manifest.json` — PWA manifest (standalone display, theme colors, SVG icon).

## Run
- Serve over HTTP (PWA requires it): `python -m http.server` then open `http://localhost:8000/`.
- Drop a `.rvf` / `.bin` / `.rvqs` / `.seed` file or scan a QR code.

## Tech stack
- Vanilla JS + WASM (no bundler). RVF format from `../../crates/rvf/`.
- Constants mirror `rvf-types/src/qr_seed.rs`.

## Related
- Sibling browser/WASM demos: `../wasm-vanilla`, `../wasm-react`, `../edge-net/dashboard`.
