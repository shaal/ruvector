# ruvector-dag/src/sona

SONA — Self-Optimising Neural Architecture for DAG learning. Requires `feature = "full"`.

- `mod.rs` — re-exports the engine, EWC++, MicroLoRA, reasoning bank, and trajectory buffer.
- `engine.rs` — `DagSonaEngine` (orchestrates adaptation across query DAGs).
- `ewc.rs` — `EwcPlusPlus`, `EwcConfig` (Elastic Weight Consolidation++ for continual learning).
- `micro_lora.rs` — `MicroLoRA`, `MicroLoRAConfig` (rank-2 LoRA adapters per operator).
- `reasoning_bank.rs` — `DagReasoningBank`, `DagPattern`, `ReasoningBankConfig` — accumulates reusable reasoning patterns.
- `trajectory.rs` — `DagTrajectory`, `DagTrajectoryBuffer` (recorded query / decision traces).

Closely mirrors the WASM-side adapter in `crates/ruvector-learning-wasm`. See `../CLAUDE.md`.
