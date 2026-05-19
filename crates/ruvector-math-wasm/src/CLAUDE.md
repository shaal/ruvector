# ruvector-math-wasm/src

JS-facing wasm-bindgen shims around `ruvector-math`.

## Files

- `lib.rs` — single source file. `#[wasm_bindgen(start)] fn start()` installs the panic hook. Defines `WasmSlicedWasserstein`
  (and other `Wasm*` wrappers) which hold `ruvector_math::*` values and expose constructors/methods callable from JS via
  `wasm-bindgen` and `serde-wasm-bindgen` for structured arg/return marshalling.
