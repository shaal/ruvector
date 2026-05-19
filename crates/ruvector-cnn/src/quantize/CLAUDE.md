# ruvector-cnn/src/quantize

INT8 quantization pipeline: calibration, graph rewrite, params, and tensor quantization. See `docs/INT8_QUANTIZATION_DESIGN.md` for the design.

## Files

- `mod.rs` — module entry.
- `params.rs` — `QuantParams` (scale + zero point) value object.
- `tensor.rs` — quantize/dequantize tensor helpers.
- `calibration.rs` — calibration over a sample dataset to derive params.
- `graph_rewrite.rs` — rewrites an FP graph to its INT8 equivalent (matches `examples/graph_rewrite_demo.rs`).
