# ruvector-rabitq-wasm

WASM bindings for `ruvector-rabitq` — 1-bit quantized vector index for browsers and edge runtimes (Cloudflare Workers, Deno, Bun). Single-threaded (the underlying `from_vectors_parallel` falls back to sequential on wasm32; output is bit-identical because rotation is deterministic).

## Layout

- `Cargo.toml` — `crate-type = ["cdylib", "rlib"]`, deps: `ruvector-rabitq`, `wasm-bindgen`, `js-sys`, `serde`, `serde-wasm-bindgen`. Default feature `console_error_panic_hook`. Release: `opt-level = "s"`, LTO. `wasm-opt = false` (handled by wasm-pack).
- `build.sh` — shell script for `wasm-pack build` invocation.
- `src/lib.rs` — sole source. Wraps `RabitqPlusIndex` from `ruvector-rabitq` and exposes a JS-friendly `RabitqIndex` class.

## Public JS API surface

- `RabitqIndex.build(vectors: Float32Array, dim, seed, rerank_factor)` — construct index.
- `idx.search(query: Float32Array, k)` — returns `Array<{id: u32, distance: f32}>`.
- `SearchResult` struct (id + approximate L2² distance after rerank) — mirrors the Python SDK shape so callers porting code get identical schemas.
- `init()` — `#[wasm_bindgen(start)]` panic-hook setup.

## Tests

- Dev-dep `wasm-bindgen-test`; no separate tests folder.

## Related crates

- `crates/ruvector-rabitq` — the native 1-bit quantized index.
