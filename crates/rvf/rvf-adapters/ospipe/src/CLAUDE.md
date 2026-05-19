# rvf-adapter-ospipe/src

Source.

## Files

- `lib.rs` — public re-exports + segment-mapping docs.
- `observation_store.rs` — `RvfObservationStore` and `ObservationMeta` (UUID→u64 mapping, field-based metadata translation).
- `pipeline.rs` — `RvfPipelineAdapter` + `PipelineConfig` driving the ingest pipeline into the store.
