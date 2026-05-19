# ruvector-nervous-system/src/hdc

Hyperdimensional Computing (HDC) primitives for neural-symbolic AI. See `HDC_IMPLEMENTATION.md` (crate root).

## Files

- `mod.rs` — façade.
- `vector.rs` — `HdcVector` type (high-dim binary/bipolar vectors).
- `ops.rs` — bind, bundle, permute operators.
- `similarity.rs` — Hamming / cosine similarity helpers.
- `memory.rs` — associative-memory item store.

Benchmarked in `benches/hdc_bench.rs`.
