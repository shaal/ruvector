# ruvllm/src/tests

In-tree (mod-style) tests that need access to crate-private items. Standalone
integration tests live in `../../tests/`.

## Files
- `mod.rs` - module wiring (gated behind `#[cfg(test)]`).
- `activation_tests.rs` - activation kernels.
- `attention_tests.rs` - attention / FlashAttention paths.
- `generation_tests.rs` - generation loop correctness.
- `gguf_tests.rs` - GGUF loader internals.
- `witness_log_tests.rs` - witness-log + semantic search internals.
