# ruvix-region

Memory region management for the RuVix Cognition Kernel (ADR-087). Regions are contiguous, capability-protected memory objects
with one of three policies: **Immutable** (set once, deduplicatable), **AppendOnly** (write cursor, bounded max_size), and
**Slab** (fixed-size slots, no fragmentation). No demand paging — all regions are physically backed at `region_map` time.

## Files

- `Cargo.toml` — depends on `ruvix-types`. Unix gets `libc` for backing memory.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.
- `benches/slab_bench.rs` — slab alloc/free throughput.
- `tests/region_test.rs` — integration tests.

## Features

- `std` (default), `mmap` (mmap-backed on Linux), `stats`.

## Public API

`RegionManager`, `RegionConfig`, region handles + backing types.
