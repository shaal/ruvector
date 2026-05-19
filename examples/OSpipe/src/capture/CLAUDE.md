# OSpipe / src / capture

Capture stage of the OSpipe pipeline. Defines the abstract `Frame` type and the capture-source surface that adapts external producers (Screenpipe, etc.) into a uniform stream feeding `../pipeline/`.

## Important files
- `mod.rs` - module root, re-exports public capture types.
- `frame.rs` - `Frame` data type and helpers describing a captured snapshot (timestamp, payload, metadata).

## Related
- Consumed by `../pipeline/ingestion.rs` and `../pipeline/dedup.rs`.
- Embeddings are produced downstream in `../storage/embedding.rs`.
