# ruvector-sparse-inference/src

Source for the sparse-inference engine.

Top-level files:
- `lib.rs` — module roots, crate docs, performance targets, public re-exports.
- `config.rs` — `SparsityConfig` etc.
- `error.rs` — `SparseInferenceError`.
- `memory.rs` — hot/cold weight cache and memory-mapped tensor helpers.
- `ops.rs` — shared compute ops invoked across backends.

Module groups (each with its own subdir CLAUDE.md):
- `backend/` — CPU / NPU / WASM SIMD compute backends.
- `model/` — GGUF / safetensors loaders and per-model runners.
- `predictor/` — low-rank P*Q neuron-activity predictor.
- `sparse/` — sparse FFN operator.
- `precision/` — quantizers, precision-lane policy, telemetry (3/5/7-bit).
- `pi/` — pi-derived calibration constants, angular embeddings, drift,
  deterministic chaos.
- `integration/` — bridges into `ruvector` and `ruvllm`.
