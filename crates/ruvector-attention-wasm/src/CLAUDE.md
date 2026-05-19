# ruvector-attention-wasm/src

WASM facade source for `ruvector-attention`.

## Files

- `lib.rs` — `#[wasm_bindgen(start)] fn init()` (panic hook), `version()`, `available_mechanisms()` returning the list of supported attention kernels.
- `attention.rs` — Per-mechanism wrappers (scaled dot product, multi-head, hyperbolic, linear, flash, local-global, MoE, CGT sheaf).
- `training.rs` — Training-mode hooks (backward pass, masking).
- `utils.rs` — Shared serialization/tensor helpers.
