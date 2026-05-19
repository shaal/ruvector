# delta-behavior / wasm

TypeScript / WebAssembly SDK on top of the Rust delta-behavior crate. Published as `@ruvector/delta-behavior`. Provides ergonomic JS wrappers over the `wasm-bindgen` exports defined in `../src/wasm.rs`.

## Important files
- `package.json` - npm package metadata; scripts `build`, `build:wasm`, `build:all`, `test`, `example:node`. Uses `tsup` for ESM+CJS bundling and `wasm-pack` for the WASM build.
- `tsconfig.json` - TypeScript compiler config.
- `tsup.config.ts` - `tsup` bundler config (entries: `src/index.ts`, `src/types.ts`).
- `example.js` - quick smoke-test script for the built package.
- `src/` - TypeScript sources (`index.ts`, `types.ts`).
- `examples/` - usage examples (`browser-example.html`, `node-example.ts`).
- `dist/` - tsup output (committed): `index.cjs`/`.js`, `types.cjs`/`.js`, and matching `.d.ts` / sourcemap files.

## Build / run
- `npm install && npm run build:all` (builds wasm + JS).
- `npm run example:node`.
- `npm test` (Vitest).

## Related
- Rust binding source: `../src/wasm.rs`. Architecture doc: `../research/WASM-DELTA-ARCHITECTURE.md`. Helper script: `../scripts/build-wasm.sh`.
