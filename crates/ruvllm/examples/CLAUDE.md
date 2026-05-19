# ruvllm/examples

Runnable examples. Each is `cargo run --example <name>`.

## Files
- `benchmark_model.rs` - benchmark a loaded model.
- `download_test_model.rs` - pull a small test model from HF Hub for
  smoke tests.
- `run_eval.rs` - run the evaluation harness (`src/evaluation/`).
- `train_contrastive.rs` - train a contrastive embedding head
  (`src/training/contrastive.rs`).
- `generate_claude_dataset.rs` - generate a Claude Flow task-routing
  dataset (`src/training/claude_dataset.rs`).
- `hub_cli.rs` - mini CLI exercising the HF Hub integration
  (`src/hub/`).
