# ruvector-dither/src

Source for the deterministic low-discrepancy dither library.

## Files

- `lib.rs` — crate docs and re-exports.
- `golden.rs` — `GoldenRatioDither::new(state)` and `.next()`; uses `frac(state + phi)` for best 1-D equidistribution.
- `pi.rs` — `PiDither::new(idx)`; iterates a precomputed table of pi bytes (period 256).
- `quantize.rs` — `quantize_dithered(value, bits, epsilon_lsb, dither)` — the core sub-LSB-offset quantizer.
- `channel.rs` — channel-aware helpers / batched application.
