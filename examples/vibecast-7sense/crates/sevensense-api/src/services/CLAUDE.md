# sevensense-api/src/services

Per-bounded-context service wiring that the REST/GraphQL/WebSocket layers call into.

## Files
- `mod.rs` - Exports each service.
- `audio.rs` - Wraps `sevensense-audio` ingestion + segmentation.
- `embedding.rs` - Wraps `sevensense-embedding` Perch 2.0 inference.
- `vector.rs` - Wraps `sevensense-vector` HNSW / Qdrant search.
- `cluster.rs` - Wraps `sevensense-analysis` clustering and motif detection.
- `interpretation.rs` - Wraps `sevensense-interpretation` LLM-based report generation.

## Related
- API handlers: `../rest/handlers.rs`, `../graphql/schema.rs`.
