# ruvLLM / src

Rust source for the top-level ruvLLM crate (host-side, not embedded).

## Top-level files
- `lib.rs` - crate root (`name = "ruvllm"`, `cdylib + rlib`).
- `config.rs` - TOML configuration loader (see `../config/example.toml`).
- `error.rs`, `types.rs` - shared error / domain types.
- `attention.rs`, `simd_inference.rs` - attention kernels and SIMD inference paths.
- `inference.rs`, `inference_real.rs` - mock and real (Candle-backed) inference engines.
- `embedding.rs` - embedding generation.
- `router.rs` - query / agent router.
- `memory.rs` - LRU / vector memory.
- `compression.rs` - tensor compression helpers.
- `orchestrator.rs` - high-level orchestrator that wires everything together.
- `learning.rs`, `training.rs` - learning and pretraining helpers.
- `napi.rs` - Node-API bindings (feature `napi`).

## Subdirectories
- `bin/` - binaries (`ruvllm-demo`, `-server`, `-bench`, `-benchmark-suite`, `-simd-demo`, `-pretrain`, `-export`).
- `sona/` - SONA continual-learning subsystem (LoRA Ultra, EWC++, ReasoningBank, trajectories, learning loops).

## Build
- `cargo build -p ruvllm --features full` (server + real-inference + hf-export + parallel + storage + metrics).
