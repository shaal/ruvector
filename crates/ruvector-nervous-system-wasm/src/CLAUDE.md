# ruvector-nervous-system-wasm/src

Source for the bio-inspired neural-system WASM bindings. Each file is one
component; `lib.rs` glues them together.

## Files
- `lib.rs` - module declarations and the `#[wasm_bindgen]` surface.
- `btsp.rs` - `BTSPLayer`: one-shot associative learning via gradient
  normalization.
- `hdc.rs` - `Hypervector`, `HdcMemory`: 10000-bit binary hypervectors,
  XOR `bind`, Hamming-distance `similarity`.
- `wta.rs` - `WTALayer` (single-pass argmax) and `KWTALayer` (partial sort
  top-k).
- `workspace.rs` - `GlobalWorkspace`, `WorkspaceItem`: 4-7 item attention
  bottleneck with broadcast competition.
