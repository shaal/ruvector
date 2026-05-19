# delta-behavior / tests

Integration tests for the delta-behavior crate.

## Important files
- `edge_cases.rs` - edge-case tests for coherence bounds, transitions, attractors.
- `edge_cases.rs.disabled` - parked variant currently excluded from compilation; rename to re-enable.
- `wasm_bindings.rs` - tests for the `#[wasm_bindgen]` surface in `../src/wasm.rs` (run with `wasm-pack test --node ../`).

## Run
- Native: `cargo test -p delta-behavior --features full`.
- WASM: `wasm-pack test --node ../`.
