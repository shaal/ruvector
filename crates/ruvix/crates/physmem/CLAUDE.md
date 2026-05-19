# ruvix-physmem

Buddy allocator for physical page-frame allocation, part of the RuVix Cognition Kernel (ADR-087). Manages physical memory using
power-of-two block sizes (4KB single page through 2MB / order 9) with minimal fragmentation.

## Files

- `Cargo.toml` — depends on `ruvix-types`. Dev: proptest, criterion.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.
- `benches/buddy_bench.rs` — alloc/free throughput at varying orders.
