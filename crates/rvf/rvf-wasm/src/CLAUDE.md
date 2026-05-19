# rvf-wasm/src

Source for the Cognitum-tile microkernel.

## Files

- `lib.rs` — `no_std` shim; 14 `#[no_mangle]` WASM exports for the core query path.
- `alloc_setup.rs` — `dlmalloc` global allocator setup (only used incidentally; static layout dominates).
- `bootstrap.rs` — read configuration from data memory and initialise the tile.
- `memory.rs` — static layout helpers (offsets, sizes).
- `store.rs` — minimal in-tile vector store.
- `segment.rs` — segment parsing over in-memory RVF data.
- `distance.rs` — fast inner-product / cosine distance kernels.
- `topk.rs` — top-k selection over distance results.
