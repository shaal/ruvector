# ruvllm/src/hub

HuggingFace Hub integration for RuvLTRA model management: upload, download,
registry, progress reporting, integrity verification.

## Files
- `mod.rs` - public API + usage docs.
- `registry.rs` - `RuvLtraRegistry` (curated list of RuvLTRA models with
  default quantizations).
- `download.rs` - `ModelDownloader` (resume-capable, integrity-checked).
- `upload.rs` - push GGUF files + SONA weights to HF Hub.
- `model_card.rs` - model-card generation/parsing.
- `progress.rs` - `indicatif`-style progress bars and reporting.
