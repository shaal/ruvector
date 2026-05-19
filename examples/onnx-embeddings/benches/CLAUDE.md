# onnx-embeddings/benches

Criterion benchmarks.

## Files

- `embedding_benchmark.rs` - End-to-end ONNX embedding throughput/latency.
- `gpu_benchmark.rs` - GPU backend benchmarks; requires `--features gpu`.

## How to run

```bash
cargo bench --bench embedding_benchmark
cargo bench --features gpu --bench gpu_benchmark
```

## Related

- GPU code: `../src/gpu/`.
- Docs: `../docs/GPU_ACCELERATION.md`.
