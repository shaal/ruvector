# ruvector-learning-wasm

MicroLoRA WASM — ultra-fast rank-2 Low-Rank Adaptation for edge AI. Designed for real-time per-operator learning in query-optimisation systems with <100 us adaptation latency, no_std-friendly, minimal allocations. Published as `@ruvector/learning-wasm` (see `pkg/package.json`).

## Architecture

```
Input (d) -> A (d x 2) -> B (2 x d) -> Delta W = alpha * (A @ B) -> Output = Input + Delta W
```

## Features

- `default = ["std"]`
- `serde = [dep:serde, dep:serde-wasm-bindgen]`
- `simd` — opt-in SIMD when available.

## Layout

- `Cargo.toml` — `cdylib` + `rlib`; ultra-aggressive size opts (`opt-level = "z"`, LTO, strip, `panic = "abort"`).
- `src/` — four files; see `src/CLAUDE.md`.
- `pkg/` — wasm-pack output (artifacts checked in for distribution); see `pkg/CLAUDE.md`.

## Public API

`LoRAConfig`, `LoRAPair`, `MicroLoRAEngine` (`lora`); `OperatorScope`, `ScopedLoRA` (`operator_scope`); `Trajectory`, `TrajectoryBuffer`, `TrajectoryStats` (`trajectory`). JS bindings re-exported via `lora::wasm_exports::*` and `operator_scope::wasm_exports::*`.

## Related

- `crates/ruvector-dag/src/sona/micro_lora.rs` — host-side MicroLoRA used inside SONA.
- `crates/micro-hnsw-wasm`, `crates/ruvector-mincut-gated-transformer-wasm` — sibling tiny WASM modules.
