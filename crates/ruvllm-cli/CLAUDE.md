# ruvllm-cli

CLI for ruvllm model management and inference on Apple Silicon. Binary name:
`ruvllm`. Wraps the `ruvllm` library plus HuggingFace Hub for download,
quantization, chat, benchmarking, and an HTTP serving mode.

## Important files
- `Cargo.toml` - `[[bin]] name = "ruvllm"` at `src/main.rs`. Pulls
  `ruvllm` (path dep) with `candle` feature. Notable deps: `clap`,
  `tokio (full + signal)`, `hf-hub` (rustls-tls; avoids openssl-sys for
  cross-build), `axum` (with `ws`), `tower-http`, `dialoguer`,
  `rustyline`, `colored`, `dirs`, `prettytable-rs`, `async-stream`.
- `src/main.rs` - clap `Cli` + `Commands` enum; tracing setup.
- `src/models.rs` - model identifier resolution + aliases (qwen, mistral,
  phi, llama).
- `src/commands/` - one file per subcommand.

## Subcommands
- `download` (alias `dl`) - pull a model (HF ID or alias) with chosen
  quantization (q4k / q8 / f16 / none).
- `list` - list available / downloaded models.
- `info <model>` - show model info.
- `serve <model>` - start the axum HTTP inference server.
- `chat <model>` - interactive chat REPL.
- `benchmark <model>` - performance benchmarks.
- `quantize <model>` - convert to GGUF format.

## Related
- `../ruvllm` - the underlying LLM runtime.
- `../sona`, `../ruvector-core`.
