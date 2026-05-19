# ruvector-wasm/tests

Browser-side integration tests for the WASM VectorDB bindings.

## Files
- `wasm.rs` - `wasm-bindgen-test` suite covering CRUD, search, batch ops,
  and (when enabled) the IndexedDB persistence path. Run via
  `wasm-pack test --headless --firefox` or similar.
