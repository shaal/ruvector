# sona (ruvector-sona)

Self-Optimizing Neural Architecture — runtime-adaptive learning for LLM routers using two-tier LoRA (micro + base), EWC++ (Elastic Weight Consolidation) to prevent catastrophic forgetting, and ReasoningBank for pattern extraction + similarity search.

The crate is named `ruvector-sona` in Cargo (published as such); the directory is `sona/`.

Three temporal learning loops:
- **Loop A (Instant)**: per-request trajectory recording + micro-LoRA updates.
- **Loop B (Background)**: hourly pattern extraction + base-LoRA updates.
- **Loop C (Deep)**: weekly dream consolidation + full EWC++ update.

## Important files

- `Cargo.toml` — `crate-type = ["cdylib", "rlib"]`. Features: `default = ["serde-support"]`, `wasm`, `napi`, `serde-support`. Disables `wasm-opt` in wasm-pack release profile.
- `LICENSE-MIT`, `LICENSE-APACHE` — Dual license.
- `BUILD_INSTRUCTIONS.md`, `WASM_COMPLETION_SUMMARY.md` — Build notes.
- `src/lib.rs` — Crate root. Module declarations + quick-start doctest demonstrating `SonaEngine::new`, `begin_trajectory`, `apply_micro_lora`.

## Source modules (`src/`)

- `engine.rs` — `SonaEngine` main entry.
- `types.rs` / `mod.rs` — Public types.
- `trajectory.rs` — Trajectory builder (input + steps + outcome).
- `lora.rs` — Micro-LoRA and Base-LoRA implementations.
- `ewc.rs` — Elastic Weight Consolidation (EWC++).
- `reasoning_bank.rs` — Pattern extraction + similarity search.
- `time_compat.rs` — Cross-target time abstraction (std/wasm).
- `wasm.rs` — `wasm-bindgen` exports (feature-gated `wasm`).
- `napi.rs`, `napi_simple.rs` — NAPI-RS exports (feature-gated `napi`).
- `loops/` — Three temporal learning loops (instant, background, coordinator).
- `training/` — Templated training pipelines (factory, federated, metrics, pipeline, templates).
- `export/` — HuggingFace export (safetensors LoRA adapters, JSONL datasets, preference pairs, distillation targets).

## Tests / Benches / Examples

- `benches/sona_bench.rs` — Criterion benches.
- `wasm-example/` — Browser HTML + `package.json` demo.

## Public API

- `SonaEngine`, `SonaConfig`.
- `SonaEngine::begin_trajectory(input) -> TrajectoryBuilder`.
- `SonaEngine::end_trajectory(builder, score)`.
- `SonaEngine::apply_micro_lora(input, output)`.

## Build

```
# Native
cargo build -p ruvector-sona
# WASM
wasm-pack build crates/sona --target web --features wasm
# NAPI (Node)
napi build --features napi
```

## Related

- Used by `ruvector-postgres/src/sona/` and `ruvector-postgres/src/dag/` for learned query optimization.
- Companion routing crate: `ruvector-tiny-dancer-core`.
- ReasoningBank pattern: also see `ruqu-exotic::reasoning_qec`.
