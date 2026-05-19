# rvf-quant

Temperature-tiered vector quantization for RVF:

| Tier | Quantization | Compression |
|------|--------------|-------------|
| Hot  | Scalar (int8) | 4× |
| Warm | Product (PQ)  | 8–16× |
| Cold | Binary (1-bit) | 32× |

A Count-Min Sketch tracks per-block access frequency to drive promotion/demotion decisions. `no_std` compatible.

## Layout

- `Cargo.toml` — features `default = ["std"]`, `std`, `simd`. Dev: `rand`, `approx`.
- `src/lib.rs` — module decls + (likely) public re-exports of the encoders.
- `src/scalar.rs` — int8 scalar quantization (hot tier).
- `src/product.rs` — Product Quantization (warm tier).
- `src/binary.rs` — 1-bit binary quantization (cold tier).
- `src/sketch.rs` — Count-Min Sketch for per-block access frequency.
- `src/tier.rs` — tier selection + promotion/demotion logic.
- `src/codec.rs` — segment codec for quantized vectors.
- `src/traits.rs` — `Quantizer` / `Dequantizer` abstractions.

## Related

- `../rvf-types` — base data-type enums
- `../rvf-runtime` — consumes these encoders during writes
- `../rvf-index` — quantization can reduce distance-comp footprint
