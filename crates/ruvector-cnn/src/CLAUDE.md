# ruvector-cnn/src

CNN crate source root.

## Top-level files

- `lib.rs` — crate doc + module wiring (`error`, `tensor` private; `kernels`, `layers`, `simd`, `int8`, `quantize` public; `backbone` behind feature).
- `embedding.rs` — `CnnEmbedder` and `MobileNetEmbedder` public surface.
- `config.rs` — `EmbeddingConfig` (dim, backbone choice, quantization toggle).
- `tensor.rs` — internal tensor representation.
- `error.rs` — crate error.

## Subdirectories

- `backbone/` — MobileNet-V3 backbone (behind `backbone` feature).
- `layers/` — full FP + quantized layer set (conv, depthwise, linear, batchnorm, activation, pooling, residual).
- `kernels/` — INT8 convolution kernels with per-arch backends.
- `simd/` — generic FP SIMD kernels (AVX2 / NEON / WASM / scalar) plus Winograd.
- `int8/` — INT8 forward-pass kernels (scalar + SIMD).
- `quantize/` — calibration, graph-rewrite pass, params, tensor quantization.
- `contrastive/` — augmentation + InfoNCE + triplet losses for contrastive training.
