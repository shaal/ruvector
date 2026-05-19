# ruvector-decompiler/tests

Integration tests covering the full decompile pipeline.

- `integration.rs` — basic end-to-end pipeline coverage.
- `ground_truth.rs` — checks decompiled output against known ground-truth corpora.
- `real_world.rs` — real-world bundle samples (regression coverage).
- `model_decompiler.rs` — LLM weight decompiler (feature `model`) covering GGUF
  and safetensors paths.
