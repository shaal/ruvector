# prime-radiant/wasm

`prime-radiant-advanced-wasm` subcrate (standalone workspace): WASM bindings for the Prime-Radiant Advanced Math modules. Exposes category, HoTT, spectral, and causal engines to JavaScript via `wasm-bindgen`. Optional `parallel` feature uses `wasm-bindgen-rayon`.

## Files

- `Cargo.toml` - Standalone package (own `[workspace]`); `cdylib`+`rlib`; release uses `opt-level="s"`, full LTO, `wasm-opt -Os`.
- `Cargo.lock` - Lockfile.
- `src/lib.rs` (~67KB) - Single-file binding implementation (the wasm crate inlines engines pending publication of the host crate).
- `pkg/` - Generated artifacts and an example TS consumer.

## How to build

```bash
cd /home/user/ruvector/examples/prime-radiant/wasm
wasm-pack build --target web --release
# or with parallel:
wasm-pack build --target web --release -- --features parallel
```

## Tech stack

- Rust 2021 + `wasm-bindgen`, `js-sys`, `web-sys`, `serde-wasm-bindgen`, `getrandom (js)`.
- Optional: `rayon`, `wasm-bindgen-rayon`.

## Related

- Native crate: `../` (`prime-radiant-category`).
- Other WASM examples: `examples/onnx-embeddings-wasm`, `examples/wasm/ios`, `examples/scipix/src/wasm`.
