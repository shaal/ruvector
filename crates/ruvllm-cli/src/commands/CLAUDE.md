# ruvllm-cli/src/commands

One file per `ruvllm` subcommand.

## Files
- `mod.rs` - re-exports each command module.
- `download.rs` - `download` (alias `dl`): pull a model from HF Hub with
  selected quantization (q4k / q8 / f16 / none).
- `list.rs` - `list`: show available / downloaded models.
- `info.rs` - `info`: show metadata for a model.
- `serve.rs` - `serve`: axum HTTP inference server (websocket capable).
- `chat.rs` - `chat`: interactive REPL using `rustyline`.
- `benchmark.rs` - `benchmark`: throughput / latency benchmarks.
- `quantize.rs` - `quantize`: convert to GGUF format.
