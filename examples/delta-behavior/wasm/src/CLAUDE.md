# delta-behavior / wasm / src

TypeScript source for the `@ruvector/delta-behavior` SDK. Wraps the `wasm-bindgen` exports from `../../src/wasm.rs` with ergonomic, typed JS APIs covering all 10 delta-behavior applications.

## Important files
- `index.ts` - main entry point; high-level classes/functions that wrap the WASM module.
- `types.ts` - shared TypeScript types (`Coherence`, `CoherenceBounds`, `DeltaConfig`, `EnergyConfig`, ...) re-exported via the package's `./types` subpath.

## Build
- From `../`: `npm run build` (or `npm run build:wasm` first, then `npm run build`).

## Related
- Rust counterpart: `../../src/wasm.rs`. Built artifacts: `../dist/` and `../pkg/` (latter created by `wasm-pack`).
