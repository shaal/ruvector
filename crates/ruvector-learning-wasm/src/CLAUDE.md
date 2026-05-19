# ruvector-learning-wasm/src

- `lib.rs` — crate root; declares the three private modules and re-exports their public types + the per-module `wasm_exports` JS bindings.
- `lora.rs` — `LoRAConfig`, `LoRAPair` (the `A: d x 2` + `B: 2 x d` matrices), `MicroLoRAEngine`; plus a `wasm_exports` submodule with `#[wasm_bindgen]` bindings.
- `operator_scope.rs` — `OperatorScope`, `ScopedLoRA` — scope an adapter to a particular operator type so different operators learn independently. Includes `wasm_exports`.
- `trajectory.rs` — `Trajectory`, `TrajectoryBuffer`, `TrajectoryStats` — record reward / loss traces feeding the engine.

See `../CLAUDE.md`.
