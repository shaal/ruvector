# ruvllm-wasm/src/workers

Web Worker pool + cross-worker messaging. Lets inference run off the main thread.

## Files

- `mod.rs` — module entry.
- `pool.rs` — worker pool (size, spawn, dispatch).
- `messages.rs` — message types passed between main thread and workers.
- `shared.rs` — shared state (SharedArrayBuffer-style helpers).
- `feature_detect.rs` — runtime detection of Worker / SharedArrayBuffer / WebGPU availability.
