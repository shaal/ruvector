# wasm/ios/src

Rust source for `ruvector-ios-wasm`.

## Files

- `lib.rs` (~36 KB) - Public API and `wasm_bindgen` exports (gated by `browser` feature).
- `hnsw.rs` (~23 KB) - HNSW index implementation.
- `distance.rs` - Distance metrics (cosine, dot, euclidean).
- `quantization.rs` - Vector quantization (e.g. PQ / scalar).
- `simd.rs` - SIMD kernels (gated by `simd` feature).
- `embeddings.rs` - Embedding helpers.
- `attention.rs` - Attention primitive.
- `qlearning.rs` - On-device Q-learning.
- `ios_capabilities.rs` - Detects iOS capabilities (Neon, BNNS, MetalPerformanceShaders).
- `ios_learning.rs` (~70 KB) - Full on-device learning loop tuned for iOS constraints.

## Related

- Benches: `../benches/`.
- Tests: `../tests/engine_tests.rs`.
- Swift consumer: `../swift/`.
