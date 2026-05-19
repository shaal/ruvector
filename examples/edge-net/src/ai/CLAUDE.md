# edge-net/src/ai

AI subsystem: attention kernels, LoRA adapters, federated training, memory, router, and the Sona reasoning bank.

## Important files
- `mod.rs` — module entry.
- `attention.rs` + `attention_unified.rs` — attention kernels.
- `dag_attention.rs` — Neural DAG attention.
- `lora.rs` — LoRA adapter logic (pairs with `../../pkg/models/microlora.js`).
- `federated.rs` — federated learning paths.
- `memory.rs` — in-WASM memory store.
- `router.rs` — request/model router.
- `sona/` — Sona reasoning bank submodule.
