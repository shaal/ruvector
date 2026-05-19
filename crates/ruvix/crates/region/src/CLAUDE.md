# ruvix-region/src

Region manager implementation.

## Files

- `lib.rs` — crate root, re-exports `RegionManager`/`RegionConfig`.
- `manager.rs` — `RegionManager`: top-level facade owning per-policy stores.
- `immutable.rs` — Immutable region implementation (set-once, deduplicatable).
- `append_only.rs` — AppendOnly region with write cursor + bounded max_size.
- `slab.rs` — Slab region: fixed-size slots from a free list, no fragmentation.
- `slab_optimized.rs` — cache-friendly slab layout variant.
- `backing.rs` — backing-memory abstraction (heap / mmap on Unix).
