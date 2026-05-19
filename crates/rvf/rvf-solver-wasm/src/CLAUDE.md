# rvf-solver-wasm/src

Source.

## Files

- `lib.rs` — `no_std` shim and `#[no_mangle] pub extern "C" fn` WASM exports (alloc/free/create/destroy/train/acceptance/result_len/result_read).
- `alloc_setup.rs` — global allocator setup using `dlmalloc`.
- `engine.rs` — three-loop adaptive solver (fast / medium / slow) with speculative dual-path execution.
- `policy.rs` — `PolicyKernel` with Thompson Sampling and 18-bucket context bandit.
- `types.rs` — shared types: puzzles, context buckets, training/holdout results.
