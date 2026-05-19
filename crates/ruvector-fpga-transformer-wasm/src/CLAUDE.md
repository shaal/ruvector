# ruvector-fpga-transformer-wasm/src

Single-file WASM binding shim.

- `lib.rs` — pulls in `ruvector_fpga_transformer::ffi::wasm_bindgen::{WasmEngine,
  micro_shape, validate_artifact}`, re-exports them as `WasmEngine`, `microShape`,
  `validateArtifact`, and defines `#[wasm_bindgen(start)] fn init()` that installs
  `console_error_panic_hook` when its feature is enabled.

All transformer logic lives in the upstream `ruvector-fpga-transformer` crate.
