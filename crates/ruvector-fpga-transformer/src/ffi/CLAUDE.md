# ruvector-fpga-transformer/src/ffi

Foreign-function interfaces for embedding the engine into non-Rust hosts.

## Files

- `mod.rs` — module entry.
- `c_abi.rs` — C ABI for embedding in C / C++ / Python.
- `wasm_bindgen.rs` — wasm-bindgen surface (gated behind `wasm` feature).
