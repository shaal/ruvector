# ruvector-math-wasm

WebAssembly bindings for `ruvector-math`. Exposes Sliced/Gromov Wasserstein, Sinkhorn, Fisher information / natural gradient,
product manifolds, and spherical-space utilities to JavaScript/TypeScript.

## Files

- `Cargo.toml` — `crate-type = ["cdylib", "rlib"]`. Depends on `ruvector-math` (path), `wasm-bindgen`, `js-sys`, `web-sys`,
  `serde-wasm-bindgen`, `getrandom` with the `js` feature. Optional `parallel` feature pulls in `rayon` + `wasm-bindgen-rayon`.
- `src/lib.rs` — `start()` initializer (installs `console_error_panic_hook` by default), plus `#[wasm_bindgen]` wrapper structs:
  `WasmSlicedWasserstein`, and (by inspection of imports) wrappers around `GromovWasserstein`, `SinkhornSolver`,
  `FisherInformation`, `NaturalGradient`, `ProductManifold`, `SphericalSpace`.

## Features

- `default = ["console_error_panic_hook"]` — friendlier panics in browsers.
- `parallel` — threaded WASM (`rayon` + `wasm-bindgen-rayon`).

## Build

- `wasm-pack` metadata sets `wasm-opt = false` to avoid double-optimization.

## Related

- `../ruvector-math` — underlying Rust math (optimal transport, information geometry, manifolds).
- Other WASM siblings: `../ruvector-dag-wasm`, `../ruvector-temporal-tensor-wasm`.
