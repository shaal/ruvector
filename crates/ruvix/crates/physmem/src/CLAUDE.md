# ruvix-physmem/src

## Files

- `lib.rs` — crate root; re-exports `BuddyAllocator` + supporting types.
- `allocator.rs` — `BuddyAllocator`: free_lists[10] (orders 0-9), base_addr, total_pages, stats.
- `frame.rs` — page-frame newtype + ops.
- `addr.rs` — physical-address newtype + arithmetic.
- `stats.rs` — allocation statistics (alloc/free counts, current usage).
- `error.rs` — alloc error enum (`OutOfMemory`, etc.).
