# rvf-wasm

RVF WASM Microkernel for Cognitum tiles. All exports are `#[no_mangle] pub extern "C" fn`. No allocator — memory is statically laid out in WASM linear memory. Target: `wasm32-unknown-unknown`, <8 KB after wasm-opt.

## Layout

- `Cargo.toml` — `cdylib`. Deps: `rvf-types` (no default), `rvf-crypto` (no default), `dlmalloc` (global). Release profile: `opt-level = "z"`, LTO, single codegen unit.
- `src/lib.rs` — `#![no_std]`; 14 `#[no_mangle]` WASM exports covering tile init, distance queries, topk, segment access.
- `src/alloc_setup.rs` — global allocator init (dlmalloc).
- `src/bootstrap.rs` — tile bootstrap from a configuration record placed in data memory.
- `src/memory.rs` — static memory layout helpers.
- `src/store.rs` — minimal in-tile store.
- `src/segment.rs` — segment parsing for in-memory RVF data.
- `src/distance.rs` — fast inner-product / cosine kernels.
- `src/topk.rs` — top-k selection over distance results.

## Related

- `../rvf-solver-wasm` — companion WASM module (temporal solver)
- `../rvf-types`, `../rvf-crypto` — no_std-friendly building blocks
