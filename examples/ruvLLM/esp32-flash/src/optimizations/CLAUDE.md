# ruvLLM / esp32-flash / src / optimizations

Quantization and inference-time optimizations bundled into the flashable firmware. Mirrors `../../../esp32/src/optimizations/`.

## Important files
- `mod.rs` - module root.
- `binary_quant.rs` - 1-bit / binary quantization.
- `product_quant.rs` - product quantization.
- `lookup_tables.rs` - LUT-based math.
- `pruning.rs` - structured/unstructured pruning.
- `sparse_attention.rs` - sparse-attention kernels.
- `micro_lora.rs` - ultra-small LoRA adapters.

## Related
- Sibling: `../../../esp32/src/optimizations/`. Demo: `../../../esp32/examples/optimization_demo.rs`.
