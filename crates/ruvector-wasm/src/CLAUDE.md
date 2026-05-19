# ruvector-wasm/src

WASM glue for Ruvector VectorDB + the kernel pack subsystem.

## Files
- `lib.rs` - main `#[wasm_bindgen]` surface. Wraps
  `ruvector_core::vector_db::VectorDB` (as `CoreVectorDB`), `SearchQuery`,
  `SearchResult`, `VectorEntry`, `DbOptions`, `DistanceMetric`, `HnswConfig`.
  Optional collections / filter integration behind `collections` feature.
- `kernel/` - kernel pack module (compiled only when `kernel-pack` feature
  is on). See `kernel/CLAUDE.md`.
- `indexeddb.js` - JS helpers invoked from Rust for IndexedDB persistence.
- `worker.js`, `worker-pool.js` - Web Worker scripts used for parallel
  search/insert paths.
