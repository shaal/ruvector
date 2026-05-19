# rvf-quant/src

Source.

## Files

- `lib.rs` — `no_std` shim + module decls; documents the three temperature tiers.
- `scalar.rs` — int8 scalar quantization (hot tier, 4× compression).
- `product.rs` — Product Quantization (warm tier, 8–16×).
- `binary.rs` — 1-bit binary quantization (cold tier, 32×).
- `sketch.rs` — Count-Min Sketch tracking per-block access frequency.
- `tier.rs` — tier selection and promote/demote decisions driven by the sketch.
- `codec.rs` — segment codec for quantized vectors.
- `traits.rs` — `Quantizer` / `Dequantizer` abstractions used by tiers.
