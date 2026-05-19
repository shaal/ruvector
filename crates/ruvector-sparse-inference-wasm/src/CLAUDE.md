# ruvector-sparse-inference-wasm/src

Single-file WASM facade for `ruvector-sparse-inference`.

## Files

- `lib.rs` — Declares `SparseInferenceEngine` with constructor (`new`), async `load_streaming(url, config)`, and `infer(input)`. Uses `GgufParser::parse` to load GGUF model bytes, `InferenceConfig` from JSON, and calls `SparseModel::forward_embedding`.

## Notes

- The streaming loader fetches via the browser `Response` API to handle very large GGUF models.
