# google-cloud/benchmark_results

Saved JSON outputs produced by the `gpu-benchmark` CLI for different workloads. These act as both regression baselines and inputs to the `report.rs` summarizer.

## Files
- `cuda_sim.json` - Results from the simulated CUDA path (`src/cuda.rs`).
- `distance_768d.json` - Distance/HNSW benchmark for 768-dimensional embeddings.
- `gnn_medium.json` - Medium-size GNN benchmark from `ruvector-gnn`.
- `quant_768d.json` - Quantized 768-dim vector benchmark.

## Producing new results
```
cargo run --release -- quick --dims 768 --num-vectors 100000 \
  --output benchmark_results/distance_768d.json
```

## Related
- `../src/benchmark.rs`, `../src/cuda.rs`, `../src/simd.rs`, `../src/report.rs`.
