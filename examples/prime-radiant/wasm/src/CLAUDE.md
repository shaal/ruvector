# prime-radiant/wasm/src

Single-file Rust source for the Prime-Radiant WASM bindings.

## Files

- `lib.rs` (~67KB) - Inlined engine implementations and `wasm_bindgen` exports for category, HoTT, spectral, and causal computations. Note: the host crate is not depended on directly; engines are reimplemented here for self-containedness.

## Related

- Manifest: `../Cargo.toml`.
- Native crate it mirrors: `../../src/`.
