# ruvLLM / esp32 / src / optimizations

Inference-time optimizations that make LLMs runnable on a microcontroller.

## Important files
- `mod.rs` - module root.
- `binary_quant.rs` - 1-bit / binary quantization.
- `product_quant.rs` - product quantization for weights / activations.
- `lookup_tables.rs` - lookup-table-based math (replacing multiplies).
- `pruning.rs` - structured / unstructured pruning.
- `sparse_attention.rs` - sparse-attention kernels suited to small SRAM.
- `micro_lora.rs` - ultra-small LoRA adapters (companion to ruvLLM SONA / LoRA Ultra).

## Related
- Showcase example: `../../examples/optimization_demo.rs`. Flashable companion: `../../../esp32-flash/src/optimizations/`.
