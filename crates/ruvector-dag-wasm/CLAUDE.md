# ruvector-dag-wasm

Minimal, size-optimized WASM DAG library for browser and embedded systems. Self-contained — depends only on `wasm-bindgen` and
serde; no other ruvector crates. Designed to compile to a tiny `cdylib` (~10KB smaller with optional `wee_alloc`).

## Files

- `Cargo.toml` — `crate-type = ["cdylib", "rlib"]`. Release profile is aggressively size-optimized (`opt-level = "z"`, LTO,
  `panic = "abort"`, single codegen unit, strip). `wasm-opt = false` to avoid double-optimization.
- `src/lib.rs` — entire crate: `WasmNode` (9 bytes: u32 + u8 + f32), `WasmDag` (`#[wasm_bindgen]` exposed) plus JS-callable
  constructors and DAG operations. Uses `wee_alloc` global allocator behind the `wee_alloc` feature.

## Features

- `default = []` — standard allocator.
- `wee_alloc` — swap in `wee_alloc::WeeAlloc` for ~10KB smaller WASM binary.

## Related

- Other WASM sibling crates: `../ruvector-math-wasm`, `../ruvector-temporal-tensor-wasm`.
- Heavyweight DAG counterpart (if present): `../ruvector-dag` / `../ruvector-graph`.
