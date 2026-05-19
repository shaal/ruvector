# ruvector-graph-transformer-wasm/src

Source for the WASM graph-transformer bindings.

## Files

- `lib.rs` — JS-facing surface (`JsGraphTransformer`); large rustdoc with usage examples.
- `transformer.rs` — implementation of sublinear attention, proof gates, Hamiltonian/spiking/causal/manifold/game-theoretic kernels.
- `utils.rs` — `JsValue` ↔ Rust conversions for vectors, edge lists, CSR arrays.
