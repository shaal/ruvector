# ruvector-router-wasm

WASM bindings for `ruvector-router-core` (the in-process vector database router).
Provides browser / WASI-friendly classes for distance metrics, vector entries, and
search queries.

## Layout

- `Cargo.toml` — `crate-type = ["cdylib", "rlib"]`. Depends on
  `ruvector-router-core`, wasm-bindgen, wasm-bindgen-futures, js-sys, web-sys
  (console). `[profile.release] opt-level = "z", lto = true, codegen-units = 1,
  panic = "abort"`.
- `package.json` — npm metadata.
- `src/lib.rs` — `#[wasm_bindgen]` exports including a `DistanceMetric` enum
  (Euclidean / Cosine / DotProduct / Manhattan) with `From` impls, a console_log
  macro, and JS-friendly wrappers around `CoreVectorDB`, `CoreVectorEntry`,
  `CoreSearchQuery`.

## Related

- `crates/ruvector-router-core` — native router implementation.
