# scipix/src/optimize

Optimization layer: SIMD, quantization, parallel batching, memory tuning.

## Files

- `mod.rs` - Module surface.
- `simd.rs` (~18 KB) - SIMD kernels.
- `quantize.rs` - Quantization helpers.
- `parallel.rs` - Parallel batching (rayon).
- `batch.rs` - Batch scheduler.
- `memory.rs` - Memory-pool / allocator utilities.

## Related

- Bench: `../../benches/optimization_bench.rs`.
- Docs: `../../docs/09_OPTIMIZATION.md`, `../../docs/optimizations.md`.
