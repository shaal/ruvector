# delta-behavior / src

Rust source for the `delta-behavior` library and benchmark driver.

## Important files
- `lib.rs` - library root: coherence types, transition gates, enforcement, attractor APIs.
- `simd_utils.rs` - SIMD helpers used by coherence computation.
- `wasm.rs` - `#[wasm_bindgen]` surface; compiled when targeting `wasm32`.
- `applications/mod.rs` - module exposing the feature-gated applications (see `../applications/`).
- `bin/run_benchmarks.rs` - benchmark driver binary, gated behind the `benchmarks` feature in `../Cargo.toml`.

## Build
- `cargo build -p delta-behavior --features full`.
- WASM: `wasm-pack build --target web ../` (or `../scripts/build-wasm.sh`).

## Related
- API docs: `../docs/API.md`. Theory: `../research/`. WASM SDK on top of `wasm.rs`: `../wasm/`.
