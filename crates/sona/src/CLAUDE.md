# sona/src

Source root for `ruvector-sona` (Self-Optimizing Neural Architecture).

## Top-level files

- `lib.rs` — Crate doc + module declarations + quick-start doctest.
- `mod.rs` — Additional module wiring (alongside `lib.rs`; primarily for re-export hygiene).
- `engine.rs` — `SonaEngine` — main API surface.
- `types.rs` — Public types (`SonaConfig`, etc.).
- `trajectory.rs` — Trajectory builder pattern.
- `lora.rs` — Micro-LoRA (rank 1-2) and Base-LoRA (standard rank) implementations.
- `ewc.rs` — EWC++ regularizer to prevent catastrophic forgetting.
- `reasoning_bank.rs` — Pattern extraction + nearest-neighbor pattern retrieval.
- `time_compat.rs` — Cross-target time helpers (std vs `getrandom`/wasm).
- `wasm.rs` — `wasm-bindgen` exports (feature `wasm`).
- `napi.rs` — Full NAPI-RS exports (feature `napi`).
- `napi_simple.rs` — Minimal NAPI surface.

## Submodules

- `loops/` — Three temporal learning loops.
- `training/` — Templated training pipelines + federated learning.
- `export/` — HuggingFace-compatible exports (safetensors, JSONL, preference pairs).
