# ruvector-postgres/src/distance

SIMD-optimized distance functions for vector similarity search.

- AVX-512: 16 floats/op
- AVX2: 8 floats/op
- ARM NEON: 4 floats/op
- Scalar fallback for all platforms

## Files

- `mod.rs` — Module entry; re-exports `scalar::*` and `simd::*`; holds a `OnceLock` for runtime CPU-feature detection.
- `scalar.rs` — Portable scalar implementations (L2, dot, cosine, Manhattan, Hamming).
- `simd.rs` — Hand-tuned SIMD implementations dispatched per CPU.
