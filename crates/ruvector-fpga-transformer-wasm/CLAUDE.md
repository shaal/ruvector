# ruvector-fpga-transformer-wasm

WebAssembly bindings for `ruvector-fpga-transformer`. Lets browsers and Node.js
run transformer inference with the same API as the native FPGA backend (native
simulator is used in WASM).

## Layout

- `Cargo.toml` — `crate-type = ["cdylib", "rlib"]`. Depends on
  `ruvector-fpga-transformer` with features `wasm` + `native_sim`, plus
  wasm-bindgen, js-sys, getrandom (js feature). `[profile.release] opt-level = "s",
  lto = true`.
- `src/lib.rs` — re-exports the upstream WASM engine and adds an `init()`
  panic-hook entry. Public symbols: `WasmEngine`, `microShape` (alias for
  `micro_shape`), `validateArtifact` (alias for `validate_artifact`).

## JS usage

```js
import { WasmEngine, microShape, validateArtifact }
  from 'ruvector-fpga-transformer-wasm';
const engine = new WasmEngine();
const modelId = engine.loadArtifact(new Uint8Array(await fetch('/model.rva').then(r => r.arrayBuffer())));
const result = engine.infer(modelId, tokens, mask, 256, false, 2);
```

## Related

- `crates/ruvector-fpga-transformer` — native FPGA transformer backend + WASM ffi.
