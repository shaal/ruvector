# ruvector-router-ffi

NAPI-RS bindings exposing `ruvector-router-core` to Node.js. Produces a `cdylib` that npm-distributable JS packages can load via `napi-rs`.

## Important files

- `Cargo.toml` — `crate-type = ["cdylib"]`. Depends on `ruvector-router-core` (path), `napi`, `napi-derive`, `tokio`, `chrono`. `[build-dependencies] napi-build = "2.1"`. Release profile: LTO, `opt-level = 3`, `strip = true`.
- `build.rs` — Runs `napi-build` setup at compile time.
- `package.json` — npm package metadata for the resulting addon.
- `src/lib.rs` — `#[deny(clippy::all)]` strict module. Defines `DistanceMetric` enum, `DbOptions` struct, `VectorDB` class wrapping `Arc<CoreVectorDB>` with an atomic `INSTANCE_COUNTER`.

## Public API (Node.js)

- `enum DistanceMetric { Euclidean, Cosine, DotProduct, Manhattan }`
- `interface DbOptions { dimensions, max_elements?, distance_metric?, hnsw_m?, hnsw_ef_construction?, hnsw_ef_search?, storage_path? }`
- `class VectorDB` — handle wrapping the core router.

## Build

```
npm run build   # uses @napi-rs/cli + the local build.rs
```

## Related

- Backbone: `ruvector-router-core`.
- WASM sibling exposing the same core: `ruvector-tiny-dancer-wasm` (different model surface).
