# scipix/benches

Criterion benchmark suite for scipix.

## Files

- `ocr_latency.rs` - Per-image OCR latency.
- `inference.rs` - Raw ONNX inference throughput.
- `preprocessing.rs` - Image preprocessing pipeline.
- `latex_generation.rs` - LaTeX synthesis from the AST.
- `cache.rs` - Cache (moka) hit/miss benchmarks.
- `api.rs` - End-to-end HTTP API throughput.
- `memory.rs` - Memory profiling.
- `optimization_bench.rs` - SIMD / quantization / parallelism comparisons.

## How to run

```bash
cargo bench -p ruvector-scipix --bench ocr_latency
cargo bench -p ruvector-scipix --bench api
bash /home/user/ruvector/examples/scipix/scripts/run_benchmarks.sh
```

## Related

- Implementations under `../src/`.
- Docs: `../docs/BENCHMARKS.md`, `../docs/08_BENCHMARKS.md`, `../docs/09_OPTIMIZATION.md`.
