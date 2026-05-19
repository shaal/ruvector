# ruvector-router-wasm/src

Single-file WASM glue layer.

- `lib.rs` — wraps `ruvector_router_core::{VectorDB, VectorEntry, SearchQuery,
  DistanceMetric}` for JS. Defines a JS-friendly `DistanceMetric` enum
  (Euclidean / Cosine / DotProduct / Manhattan) with `From` conversion to the
  core type, a `console_log!` helper that calls into `console.log`, and the
  exported router classes.
