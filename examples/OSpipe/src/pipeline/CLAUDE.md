# OSpipe / src / pipeline

Ingestion pipeline. Takes captured `Frame`s (from `../capture/`), deduplicates them, and feeds the embedding + storage stages.

## Important files
- `mod.rs` - module root, pipeline orchestration.
- `ingestion.rs` - end-to-end ingest stage: receive frames, dispatch to embed + store + graph.
- `dedup.rs` - deduplication (likely hash/near-hash based) to avoid storing redundant frames.

## Related
- Upstream: `../capture/`. Downstream: `../storage/`, `../graph/`.
