# ruvector-sparse-inference

PowerInfer-style sparse inference engine for edge devices. Exploits power-law
activation locality to compute only "hot" neurons; provides low-rank predictors
(P*Q factorisation), SIMD kernels (AVX2 / SSE4.1 / NEON / WASM SIMD), GGUF
loading for quantized Llama models, hot/cold caching, and a `pi`-based
calibration / drift detection subsystem (3/5/7-bit precision lanes).

Targets: LFM2 350M ~5-10 ms/sentence (2.5x), Llama 7B 50-100 ms/token (5-10x),
1.5-2x memory reduction via weight offloading.

## Layout

- `Cargo.toml` — lib. Deps: ndarray, rand, serde, rkyv, thiserror, anyhow,
  tracing, rayon, parking_lot, memmap2, byteorder, half. `[[bench]]`s:
  `sparse_inference_bench`. Dev: criterion, proptest, mockall.
- `BUILD_STATUS.md` — current build / port status.
- `src/lib.rs` — module roots, public docs, performance targets.
- `src/config.rs`, `src/error.rs`, `src/memory.rs`, `src/ops.rs` — shared types.
- `src/backend/` — per-target compute backends (`cpu.rs`, `npu.rs`, `wasm.rs`).
- `src/model/` — model loaders / runners (incl. GGUF + safetensors).
- `src/predictor/` — low-rank neuron-activity predictor.
- `src/sparse/` — sparse FFN op.
- `src/precision/` — precision-lane policy + telemetry (3/5/7-bit).
- `src/pi/` — pi-derived calibration constants, angular embeddings, drift, chaos.
- `src/integration/` — bridges to `ruvector` / `ruvllm`.
- `benches/sparse_inference_bench.rs`, `benches/simd_kernels.rs` — Criterion.
- `docs/GGUF_IMPLEMENTATION.md` — GGUF loader design notes.
- `examples/basic_usage.rs`, `examples/gguf_loader.rs`.
- `tests/` — backend SIMD tests, integration (model loading, inference), unit
  (predictor / quantization / sparse FFN), property tests, common helpers.

## Public API

`SparseInferenceEngine`, `SparsityConfig`, `PiContext`, `PrecisionLane`, plus
module-level exports.

## Related

- `crates/ruvector-sona`, `crates/ruvllm`, `crates/ruvector-core`.
