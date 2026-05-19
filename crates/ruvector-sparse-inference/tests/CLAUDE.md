# ruvector-sparse-inference/tests

Integration, unit, and property test suites.

- `backend_simd_tests.rs` — SIMD kernel parity across backends (CPU/NPU/WASM).
- `common/` — shared helpers (`mod.rs`).
- `integration/` — `model_loading_tests.rs`, `sparse_inference_tests.rs`
  (GGUF/safetensors load + end-to-end inference).
- `unit/` — `predictor_tests.rs`, `quantization_tests.rs`, `sparse_ffn_tests.rs`.
- `property/` — `mod.rs` — proptest-based invariants.
