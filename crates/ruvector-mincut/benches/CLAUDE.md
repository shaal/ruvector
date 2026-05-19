# ruvector-mincut/benches

Criterion benchmarks for the min-cut algorithm family.

## Files

- `mincut_bench.rs` — baseline insert/delete/query throughput on `DynamicMinCut`.
- `bounded_bench.rs` — bounded-instance variant performance.
- `canonical_bench.rs` — canonical decomposition micro-benchmarks.
- `jtree_bench.rs` — J-tree hierarchy benchmarks.
- `optimization_bench.rs` — cache / SIMD / parallel optimization micro-benchmarks.
- `paper_algorithms_bench.rs` — benchmarks for the published paper algorithms (arXiv:2512.13105).
- `snn_bench.rs` — spiking-neural-network cognitive-engine layer benchmarks.
- `sota_bench.rs` — combined SOTA configuration benchmark.
