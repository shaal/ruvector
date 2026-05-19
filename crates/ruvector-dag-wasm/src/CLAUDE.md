# ruvector-dag-wasm/src

Single-file WASM DAG implementation.

## Files

- `lib.rs` — defines `WasmNode` (id: u32, op: u8, cost: f32 — 9 bytes packed) and `WasmDag` (`#[wasm_bindgen]` JS-exposed class
  with `Vec<WasmNode>` + `Vec<(u32,u32)>` edges). Hot paths are inlined and avoid string ops. Globally selects `wee_alloc` when the
  `wee_alloc` feature is enabled.
