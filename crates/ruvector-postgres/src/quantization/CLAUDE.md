# ruvector-postgres/src/quantization

Vector quantization for memory reduction.

- Scalar (SQ8): 4x compression
- Product (PQ): 8-32x compression
- Binary: 32x compression

## Files

- `mod.rs` — Global `TABLE_MEMORY_BYTES: AtomicUsize` for tracking quantization-table memory; declares submodules.
- `scalar.rs` — SQ8 scalar quantization.
- `product.rs` — PQ product quantization.
- `binary.rs` — Binary/Hamming quantization.

## Pointers

- See `../../docs/QUANTIZED_TYPES.md`.
- Storage types live in `../types/` (`BinaryVec`, `ProductVec`, `ScalarVec`, `HalfVec`).
