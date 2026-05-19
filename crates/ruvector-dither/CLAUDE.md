# ruvector-dither

Deterministic, low-discrepancy pre-quantization dithering for low-bit inference on tiny devices (WASM, Seed, STM32). Provides
golden-ratio and pi-digit sequences for blue-noise error shaping with zero runtime deps and full `no_std` compatibility.

## Why

Quantizers at 3/5/7 bits align with power-of-two grid boundaries and produce idle tones / limit cycles. A sub-LSB pre-quantization
offset decorrelates the signal from grid boundaries and pushes error toward high frequencies — without RNG, so outputs are
deterministic and reproducible across WASM / x86 / ARM.

## Files

- `Cargo.toml` — zero runtime deps. Single criterion bench (`dither_bench`). Features: `no_std` (requires an allocator).
- `README.md` — public-facing docs (referenced in package metadata).
- `src/lib.rs` — crate root; re-exports the dither sequences and `quantize_dithered`.
- `src/golden.rs` — `GoldenRatioDither` (state update: `frac(state + phi)`), best 1-D equidistribution.
- `src/pi.rs` — `PiDither` (table of pi bytes, period = 256), most reproducible.
- `src/quantize.rs` — `quantize_dithered(x, bits, epsilon_lsb, dither)` core kernel.
- `src/channel.rs` — multi-channel / batched dithering helpers.
- `benches/dither_bench.rs` — criterion microbenchmarks for sequence cost and quantization throughput.

## Features

- `default = []` — std build with no extras.
- `no_std` — bare-metal / WASM target (requires external allocator).

## Public API surface

`GoldenRatioDither`, `PiDither`, `quantize_dithered`, plus channel helpers.

## Related

- `../ruvector-attention`, `../ruvector-quantization` (likely) — consumers of dithering before low-bit quantization.
