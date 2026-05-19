# ruvector-cnn/src/int8

INT8 forward-pass kernels for the quantized embedded path.

## Files

- `mod.rs` — module entry + arch dispatch.

## Subdirectories

- `kernels/` — actual kernel implementations (scalar + SIMD).

Pairs with `src/quantize/` (calibration + graph rewrite) and `src/kernels/` (the lower-level INT8 conv kernels also used by the quantized layers).
