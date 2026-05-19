# rvf-adapter-ospipe

RVF adapter for OSpipe's observation-state pipeline. Maps:

- **VEC_SEG** — state vector embeddings (screen, audio, UI observations)
- **META_SEG** — observation metadata (app name, content type, timestamps)
- **JOURNAL_SEG** — deletion records for expired observations

Bridges OSpipe's `StoredEmbedding` / `CapturedFrame` (UUIDs, chrono timestamps, JSON metadata) to RVF's u64-id + field-based metadata model.

## Layout

- `Cargo.toml` — name `rvf-adapter-ospipe`. Deps: `rvf-runtime`, `rvf-types` (`std`). Dev: `tempfile`.
- `src/lib.rs` — re-exports `ObservationMeta`, `RvfObservationStore`, `PipelineConfig`, `RvfPipelineAdapter`.
- `src/observation_store.rs` — `RvfObservationStore` + `ObservationMeta`.
- `src/pipeline.rs` — `RvfPipelineAdapter` + `PipelineConfig` for ingest pipeline.

## Related

- `../../rvf-runtime`, `../../rvf-types`
- Sibling adapters under `../`
