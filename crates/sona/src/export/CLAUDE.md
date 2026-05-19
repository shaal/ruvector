# sona/src/export

HuggingFace integration — export learned patterns, LoRA weights, and trajectories to HuggingFace-compatible formats for pretraining, fine-tuning, and knowledge distillation.

## Files

- `mod.rs` — Module entry + export-format documentation.
- `safetensors.rs` — LoRA adapter weights in PEFT-compatible `.safetensors` format.
- `dataset.rs` — JSONL dataset writer (ReasoningBank patterns -> HF dataset).
- `pretrain.rs` — Pretraining export pipeline.
- `huggingface_hub.rs` — HuggingFace Hub upload integration.

Other supported targets (per crate doc): preference pairs for DPO/RLHF, distillation targets for knowledge distillation.
