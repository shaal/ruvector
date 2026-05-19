# rvlite/tests

Integration tests for the RvLite WASM + Rust API.

## Files
- `cypher_integration_test.rs` - Cypher query end-to-end tests against
  the embedded graph store.
- `wasm.rs` - `wasm-bindgen-test` suite covering the JS-facing `RvLite`
  surface. Run via `wasm-pack test` with the appropriate browser target.
