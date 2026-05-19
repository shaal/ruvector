# @ruvector/diskann

DiskANN/Vamana SSD-friendly billion-scale approximate-nearest-neighbor index with product quantization. Native NAPI binding façade — actual binaries ship as platform-specific optional deps.

## Key files

- `package.json` — `@ruvector/diskann` v0.1.0; main `index.js`, types `index.d.ts`.
- `test.js` — smoke test (`npm test`).
- `false` — zero-byte placeholder (likely accidental from a redirect).

## Published API

Loader for native bindings (`.node` files) provided by optional deps:

- `@ruvector/diskann-linux-x64-gnu`, `-linux-arm64-gnu`, `-darwin-x64`, `-darwin-arm64`, `-win32-x64-msvc` (all v0.1.0).

Note: `index.js` / `index.d.ts` are not present in this checkout; they would be generated/added at publish time.

## Scripts

- `test` -> `node test.js`

## Related

- Rust crate: see ruvector repo (DiskANN crate / Vamana implementation).
- Sibling: `npm/packages/ruvector/` peer-depends on `@ruvector/diskann` (optional).
