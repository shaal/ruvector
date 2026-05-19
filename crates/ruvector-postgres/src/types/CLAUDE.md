# ruvector-postgres/src/types

Vector type implementations for PostgreSQL with zero-copy optimizations.

- `RuVector` — primary f32 vector type (pgvector compatible).
- `HalfVec` — half-precision (f16) for memory savings.
- `SparseVec` — sparse vector for high-dimensional data.
- `BinaryVec` — binary-quantized vectors.
- `ProductVec` — product-quantized vectors.
- `ScalarVec` — scalar-quantized vectors.

Features: zero-copy via `VectorData` trait, PostgreSQL memory-context integration, shared-memory structures for indexes, TOAST handling for large vectors.

## Files

- `mod.rs` — Module entry; type re-exports.
- `vector.rs` — `RuVector` primary type.
- `halfvec.rs` — `HalfVec` (f16).
- `halfvec_summary.md` — Design notes for `HalfVec` (markdown, not compiled).
- `sparsevec.rs` — `SparseVec`.
- `binaryvec.rs` — `BinaryVec`.
- `productvec.rs` — `ProductVec`.
- `scalarvec.rs` — `ScalarVec`.

## Pointers

- See `../../docs/QUANTIZED_TYPES.md`, `../../docs/NATIVE_TYPE_IO.md`, `../../docs/TYPE_IO_IMPLEMENTATION_SUMMARY.md`.
