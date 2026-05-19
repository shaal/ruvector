# ruvector-cnn

Pure-Rust CNN feature extraction for image embeddings with SIMD acceleration (AVX2, NEON, WASM SIMD128) and INT8 quantization. Targets CPU + WASM with no BLAS/OpenCV deps.

## Layout

- `Cargo.toml` — features: `std` (default), `wasm`, `simd`, `augmentation` (pulls in `image`), `backbone` (enables MobileNet path; requires API fixes). Deps: `serde`, `thiserror`, `nalgebra`, `rand`, `rand_distr`. Benches via `criterion` + `fastrand`. Research-tier lint allow-list.
- `src/lib.rs` — crate doc + module list; `mod error; mod tensor;` (private) + public `kernels`, `layers`, `simd`, `int8`, `quantize`, and `backbone` behind feature.
- `src/embedding.rs` — `CnnEmbedder` / `MobileNetEmbedder` public API.
- `src/config.rs` — `EmbeddingConfig`.
- `src/tensor.rs` — internal tensor type.
- `src/error.rs` — module error.

## Module groups (under `src/`)

- `backbone/` — MobileNet-V3 small/large backbone (behind `backbone` feature).
- `layers/` — conv, linear, batchnorm, activation, pooling + quantized variants.
- `kernels/` — INT8 conv kernels with per-arch backends (AVX2, NEON, WASM, scalar).
- `simd/` — generic FP SIMD kernels (AVX2, NEON, WASM, scalar, Winograd).
- `int8/` — INT8 forward pass kernels for the embedded INT8 path.
- `quantize/` — calibration, graph rewrite, params, tensor quantization.
- `contrastive/` — contrastive training utilities (InfoNCE, triplet, augmentation).

## Tests / benches / examples

- `benches/cnn_benchmarks.rs`, `benches/int8_bench.rs`.
- `examples/graph_rewrite_demo.rs`.
- `tests/`: `acceptance_gates.rs`, `backbone_test.rs`, `contrastive_test.rs`, `graph_rewrite_integration.rs`, `integration_test.rs`, `kernel_equivalence.rs`, `layers_test.rs`, `quality_validation.rs`, `simd_test.rs`.

## Docs

`docs/` holds ADR-091 phase summaries, INT8 design notes (`INT8_KERNELS_IMPLEMENTATION.md`, `INT8_QUANTIZATION_DESIGN.md`, `QUANTIZED_LAYERS_USAGE.md`), graph-rewrite notes.

## Related crates

Pairs with embedding consumers like `crates/ruvector-core` (for vector storage) and `crates/ruvector-hailo` (for NPU-backed embeddings).
