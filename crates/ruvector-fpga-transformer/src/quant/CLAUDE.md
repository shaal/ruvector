# ruvector-fpga-transformer/src/quant

Quantization formats and helpers for the FPGA path.

## Files

- `mod.rs` — module entry.
- `qformat.rs` — INT4 / INT8 format definitions.
- `calib.rs` — calibration helpers (range, scale, zero point).
- `lut.rs` — lookup tables used by `lut_softmax` / `pwl_softmax` features.
