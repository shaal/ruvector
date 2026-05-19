# ruvector-rabitq

RaBitQ: rotation-based 1-bit quantization for ultra-fast approximate
nearest-neighbor search with theoretical error bounds. Inspired by Gao &
Long, *"RaBitQ"*, SIGMOD 2024. Ships two estimators - symmetric
(Charikar-style 1-bit query) and asymmetric (RaBitQ-2024-style f32 query,
1-bit DB) - and three index variants.

## Important files
- `Cargo.toml` - workspace inheritance. `[[bin]] rabitq-demo`.
  `[[bench]] rabitq_bench` (criterion). Deps: `rand`, `rand_distr`,
  `serde`, `serde_json`, `thiserror`. `rayon` only on non-wasm targets
  (output stays bit-identical because rotation is deterministic).
- `BENCHMARK.md` - recall + throughput numbers across n in {1k, 5k, 50k,
  100k}.
- `src/lib.rs` - public surface, math notes, guarantees.
- `src/main.rs` - `rabitq-demo` binary: end-to-end recall + throughput sweep.
- `src/error.rs` - `RabitqError`.
- `src/index.rs` - `AnnIndex` trait + `FlatF32Index`, `RabitqIndex`,
  `RabitqPlusIndex` (with rerank), `RabitqAsymIndex`; `SearchResult`.
- `src/kernel.rs` - `VectorKernel` abstraction, `CpuKernel`, `KernelCaps`,
  `ScanRequest`, `ScanResponse`.
- `src/quantize.rs` - `BinaryCode`, `pack_bits`, `unpack_bits`.
- `src/rotation.rs` - `RandomRotation`, `RandomRotationKind` (Haar-uniform
  base, deterministic).
- `src/scan.rs` - scan/heap loop; bounded max-heap top-k (`f32::total_cmp`,
  no NaN panics).
- `src/persist.rs` - serialization for built indexes.
- `benches/rabitq_bench.rs` - criterion micro-benches.

## Guarantees
Padding-safe popcount at any D; deterministic across runs given
`(dim, seed, data)`; no `unsafe`; no external BLAS/LAPACK.

## Related
- Used as a quantization backend for other ANN flows in `ruvector-core` /
  `ruvector-collections`.
