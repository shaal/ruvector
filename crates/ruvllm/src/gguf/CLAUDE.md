# ruvllm/src/gguf

llama.cpp-compatible GGUF model format loader. Parses GGUF v3 with
memory-mapped support, decodes all major quantization types (Q4_0, Q4_K,
Q8_0, F16, F32, ...), and streams large tensors chunk-by-chunk.

## Files
- `mod.rs` - public API + supported quantization table.
- `parser.rs` - GGUF v3 binary parser (header, metadata, tensor index).
- `loader.rs` - memory-mapped loader + tensor materialization.
- `tensors.rs` - tensor descriptors and access helpers.
- `quantization.rs` - per-quant-type decoders (Q4_0/Q4_K/Q8_0/F16/F32/...).
- `model_init.rs` - convert a loaded GGUF into an in-memory model ready
  for inference.
