# OSpipe / tests

Integration tests for the OSpipe crate, split by target.

## Important files
- `integration.rs` - native integration tests covering the capture -> pipeline -> storage -> search flow.
- `wasm.rs` - `wasm-bindgen-test` suite for the `wasm32` build (the WASM bindings in `../src/wasm/`).

## Run
- Native: `cargo test -p ospipe --test integration`.
- WASM: `wasm-pack test --node ../` (matches the dev-dependency configuration).
