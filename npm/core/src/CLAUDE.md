# npm/core/src

TypeScript source for `@ruvector/core`. Contains both the ESM and CommonJS entry points along with their compiled `.js` / `.d.ts` / `.map` artifacts (the build pipeline runs `tsc` against `../tsconfig.json` for ESM and `../tsconfig.cjs.json` for CJS).

## Files

- `index.ts` - Primary ESM entry. Declares the public TypeScript surface (`VectorDB`, `CollectionManager`, `DbOptions`, `HnswConfig`, `QuantizationConfig`, `SearchQuery`, `SearchResult`, `VectorEntry`, `Filter`, `CollectionConfig`, `CollectionStats`, `Alias`, `HealthResponse`, `NativeBinding`) and the `DistanceMetric` enum (Euclidean / Cosine / DotProduct / Manhattan). Implements `detectPlatform()` and `loadNativeBinding()` which tries `../native/<platform-arch>/` first, then falls back to the published `@ruvector/core-<platform>` package. Also probes for an optional `@ruvector/attention` dependency.
- `index.cjs.ts` - Parallel CommonJS-compatible source compiled to `dist/index.cjs`. Resolves the `ruvector-core-<platform>` package (note: unscoped names), then falls back to `../platforms/<platform>/ruvector.node`. Re-exports the native binding directly with a `VectorDb`->`VectorDB` alias.
- `index.cjs.js`, `index.cjs.d.ts`, `*.map` - Build outputs for the CJS variant.
- `index.d.ts.map`, `index.js.map` - Source maps for the ESM build (the actual `.d.ts` / `.js` live in `../dist`).

## Notes

- NAPI-RS emits the class as `VectorDb` (lowercase d); both entry points re-export it as `VectorDB` for ergonomic JS/TS consumption.
- Native loading paths intentionally diverge: the ESM path checks `../native/.../index.cjs` then `ruvector.node`; the CJS path tries the npm package name first then `../platforms/.../ruvector.node`.

## Related

- `../tsconfig.json`, `../tsconfig.cjs.json` - Compiler configs.
- `../native/`, `../platforms/` - Targets of the runtime loader.
