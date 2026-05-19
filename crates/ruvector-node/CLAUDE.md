# ruvector-node

Node.js bindings for Ruvector via NAPI-RS. High-performance Rust vector database with zero-copy buffer sharing, async/await support, and complete TypeScript type definitions.

## Layout

- `Cargo.toml` — `crate-type = ["cdylib"]`, deps: `ruvector-core`, `ruvector-collections`, `ruvector-filter`, `ruvector-metrics`, `napi`, `napi-derive`, `tokio`, `serde`, `tracing`. Release profile uses LTO + strip.
- `package.json` — npm package metadata (TypeScript wrapper consumers).
- `build.rs` — `napi-build` invocation.
- `PHASE5_STATUS.md` — internal status doc.
- `.npmignore`, `.gitignore` — distribution exclusions.
- `src/lib.rs` — entire NAPI surface in one file: distance metric enum (`JsDistanceMetric`), config wrappers, async DB methods, collection/filter/metrics surfaces.

## Public JS API surface

- `JsDistanceMetric` enum (Euclidean / Cosine / DotProduct / Manhattan).
- Wraps `ruvector_core::VectorDB`, `SearchQuery`, `SearchResult`, `VectorEntry`, `HnswConfig`, `QuantizationConfig`, `DbOptions`.
- Adds `CollectionManager`, `FilterExpression`, `gather_metrics`, `HealthChecker`/`HealthStatus`.

## Tests / examples

- `examples/`: `simple.mjs`, `advanced.mjs`, `semantic-search.mjs` — runnable from a built npm install.
- `tests/`: `basic.test.mjs`, `benchmark.test.mjs` — JS-side tests.

## Related crates / wrappers

- Mirrors functionality of the workspace's Python (`ruvector-python`) and WASM (`ruvector-wasm`) bindings.
- Underlying crates: `ruvector-core`, `ruvector-collections`, `ruvector-filter`, `ruvector-metrics`.
