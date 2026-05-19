# ruvector-rabitq/src

Source for the RaBitQ rotation-based 1-bit ANN index. Modules form a clean
build -> persist -> search pipeline.

## Files
- `lib.rs` - public re-exports + crate-level docs (math, guarantees).
- `main.rs` - `rabitq-demo` binary: end-to-end recall + throughput sweep
  across n in {1k, 5k, 50k, 100k}.
- `error.rs` - `RabitqError` enum.
- `index.rs` - `AnnIndex` trait and the four index variants:
  `FlatF32Index` (exact baseline), `RabitqIndex` (symmetric, no rerank),
  `RabitqPlusIndex` (symmetric + exact rerank with stored originals),
  `RabitqAsymIndex` (asymmetric, optional rerank). `SearchResult` struct.
- `kernel.rs` - `VectorKernel` trait, `CpuKernel` impl, `KernelCaps`,
  `ScanRequest`, `ScanResponse`. Lets the scan loop swap in custom kernels.
- `quantize.rs` - `BinaryCode`, `pack_bits`, `unpack_bits` (padding-safe
  popcount).
- `rotation.rs` - `RandomRotation` + `RandomRotationKind` (Haar-uniform);
  deterministic from seed.
- `scan.rs` - top-k scan via bounded max-heap, `f32::total_cmp` (NaN-safe).
- `persist.rs` - serialize/deserialize built indexes via `serde_json` /
  binary.
