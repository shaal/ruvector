# ruvLLM / esp32 / src / models

Model metadata / loaders for the on-device LLMs shipped with `ruvllm-esp32`.

## Important files
- `mod.rs` - module root; defines the supported model catalog (size, quantization, layout) and the loading helpers used by `../micro_inference.rs` / `../quantized.rs`.

## Related
- Sizing examples: `../../examples/model_sizing_demo.rs`, `../../examples/medium_scale_demo.rs`, `../../examples/massive_scale_demo.rs`.
