# docs/adr/temporal-tensor-store/

ADR subseries for the **Temporal Tensor Store** subsystem - block-based, tiered, temporally-aware vector/tensor storage with delta compression. These ADRs (numbered 018..023) belong to the main series but are grouped here for cohesion. Parent: `../ADR-017-temporal-tensor-compression.md`.

## ADRs

- `ADR-018-block-based-storage-engine.md` - block-oriented storage engine.
- `ADR-019-tiered-quantization-formats.md` - tiered quantization formats per temperature.
- `ADR-020-temporal-scoring-tier-migration.md` - scoring and tier migration policy.
- `ADR-021-delta-compression-reconstruction.md` - delta compression and reconstruction.
- `ADR-022-wasm-api-cross-platform.md` - WASM API surface and cross-platform.
- `ADR-023-benchmarking-acceptance-criteria.md` - benchmarks and acceptance criteria.

## Related

- `../ADR-017-temporal-tensor-compression.md` - parent ADR.
- `../../architecture/temporal-tensor-store-ddd.md` - DDD design.
- `../delta-behavior/` - related delta encoding subseries.
