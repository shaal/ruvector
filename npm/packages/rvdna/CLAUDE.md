# @ruvector/rvdna

AI-native genomic analysis: 20-SNP biomarker risk scoring, streaming
anomaly detection, 64-dim profile vectors, 23andMe genotype parsing,
CYP2D6/CYP2C19 pharmacogenomics, variant calling, protein prediction,
and HNSW vector search. Pure-JS fallback today, with napi build wiring
ready when platform binaries are republished.

## Important files
- `package.json` - npm metadata (`@ruvector/rvdna` v0.3.0). Note: the
  inline `//optionalDependencies` comment documents that platform-
  binary optional deps were intentionally removed on 2026-05-16 to
  stop spurious install warnings — re-add when napi actually publishes
  binaries.
- `index.js` - Platform shim. Maps `process.platform`/`arch` to per-
  platform `@ruvector/rvdna-<triple>` packages, falls back to the pure
  JS shim in `src/` if the native module is not found.
- `index.d.ts` - TypeScript declarations for the exported APIs.
- `src/biomarker.js` - Pure-JS biomarker reference ranges
  (`BIOMARKER_REFERENCES`) and scoring fallback (mirror of `biomarker.rs`).
- `src/stream.js` - Streaming anomaly detection fallback.
- `tests/` - Vitest-style harness (`test-biomarker.js`,
  `test-real-data.js`) plus 23andMe fixture files.

## Exports / scripts
- `main` -> `index.js`, `types` -> `index.d.ts`. Published files:
  `index.js`, `index.d.ts`, `src/`, `README.md`.
- `build:napi` - `napi build --platform --release --cargo-cwd
  ../../../examples/dna` (note: source crate lives under examples/).
- `test` - `node tests/test-biomarker.js`.

## Related
- Rust source: `../../../examples/dna` (the cargo crate that produces
  the napi binary).
