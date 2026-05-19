# ruvllm/src/quantize

Quantization pipeline for converting full-precision models to edge-friendly
formats. Supports llama.cpp-style Q4_K_M / Q5_K_M / Q8_0 alongside PiQ3
(pi-constant ultra-low-bit quantization) and QuIP (Hadamard-based).

## Files
- `mod.rs` - public API + supported-format table.
- `pi_quant.rs` - Pi-quantization (irrational step sizes for non-uniform
  grids).
- `pi_quant_simd.rs` - SIMD-accelerated Pi-quantization kernels.
- `hadamard.rs` - Hadamard transform building block (for QuIP).
- `incoherence.rs` - incoherence processing for QuIP.
- `quip.rs` - QuIP quantization pipeline (Hadamard + incoherence).
- `ruvltra_quant.rs` - end-to-end quantizer for RuvLTRA models.
- `security.rs` - safety checks (weight-bound validation, NaN/Inf scans).
