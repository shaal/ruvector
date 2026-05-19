# ruvector-dither/benches

Criterion benchmarks for the dither sequences and quantizer.

## Files

- `dither_bench.rs` — single `harness = false` target measuring `GoldenRatioDither` and `PiDither` sequence cost plus
  `quantize_dithered` throughput at common bit widths.

Run: `cargo bench -p ruvector-dither`.
