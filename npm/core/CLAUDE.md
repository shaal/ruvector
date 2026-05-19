# npm/core

Source for the `@ruvector/core` npm package: a TypeScript wrapper around the Rust NAPI-RS native bindings that ships the high-performance vector database (HNSW + SIMD) to Node.js. It auto-detects the host platform/architecture and loads the appropriate prebuilt `.node` binary from a sibling `@ruvector/core-<platform>` package (or from the local `native/` / `platforms/` directories during development).

## Important files

- `package.json` - `@ruvector/core` v0.1.17, ESM-first with CJS fallback. Optional dependency on `@ruvector/attention`. `node >= 18`.
- `src/index.ts` - Main ESM entry. Defines TypeScript interfaces (`VectorDB`, `CollectionManager`, `DbOptions`, `HnswConfig`, `QuantizationConfig`, `SearchQuery`, `Filter`, etc.), the `DistanceMetric` enum, and the platform-detection / native-binding loader.
- `src/index.cjs.ts` - CommonJS-compatible wrapper compiled with `tsconfig.cjs.json`. Renamed to `dist/index.cjs` during build.
- `tsconfig.json` - ESM build (ES2022, Node16 module resolution, strict). Outputs to `./dist`.
- `tsconfig.cjs.json` - CJS build for `index.cjs.ts`, outputs to `./dist-cjs` then renamed.
- `test-binding.mjs`, `test-native.mjs`, `test-package.cjs` - Hand-run smoke tests verifying the native binding loads and basic ops work in ESM/CJS contexts.
- `.npmignore`, `LICENSE` (MIT) - Publishing metadata.
- `native/` - Locally-built `.node` binaries used during development (currently only `linux-x64`).
- `platforms/` - Per-platform sibling npm packages that ship the prebuilt `.node` binaries.

## Exports

`VectorDB`, `CollectionManager`, `DistanceMetric`, `version`, `hello`, `getMetrics`, `getHealth`, and an optional `attention` re-export. Re-exports type interfaces for the full API.

## Scripts

- `npm run build` - Build ESM + CJS, then rename CJS output.
- `npm run build:esm` / `npm run build:cjs` - Individual TypeScript builds.
- `npm test` - `node --test` (uses the hand-written `test-*.mjs` / `.cjs` scripts when run directly).
- `npm run clean` - Remove `dist` and `dist-cjs`.

## Related dirs

- `../../crates/ruvector-core` - The underlying Rust crate.
- The NAPI-RS Rust binding crate produces the `.node` artifacts placed in `native/<platform>/` and `platforms/<platform>/`.
- `../wasm` - Sibling WebAssembly build for browsers / pure-JS fallback.
- `../packages/` - Higher-level packages (e.g. `ruvector` meta-package, attention, CLI) that depend on `@ruvector/core`.
