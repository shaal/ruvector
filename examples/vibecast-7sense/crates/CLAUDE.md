# vibecast-7sense/crates

Nine domain crates that make up the 7sense bioacoustics platform, organized along DDD bounded contexts.

## Crates
- `sevensense-core/` - Foundational types: strongly-typed IDs, value objects (GeoLocation, Timestamp, AudioMetadata), errors, domain entities/events.
- `sevensense-audio/` - Audio decoding (WAV/FLAC/MP3/Ogg via Symphonia), resampling, segmentation, spectrograms.
- `sevensense-embedding/` - Perch 2.0 ONNX inference for 1536-dim bioacoustic embeddings (via `ort`).
- `sevensense-vector/` - HNSW indexing, Qdrant client wrapper, hyperbolic embeddings, collection management.
- `sevensense-learning/` - GNN (GCN/GraphSAGE/GAT), contrastive learning, Elastic Weight Consolidation, attention.
- `sevensense-analysis/` - HDBSCAN/k-means clustering, motif detection, Markov sequence analysis, anomaly detection.
- `sevensense-interpretation/` - LLM-based report generation, conservation insights, prompt templates.
- `sevensense-api/` - REST + GraphQL + WebSocket server (Axum).
- `sevensense-benches/` - Workspace benchmarking utilities.

## Dependency direction
`core` <- everyone; `audio`/`embedding` -> `core`; `vector` -> `core`; `learning`/`analysis` -> `core,vector`; `interpretation` -> `core,analysis`; `api` -> all of the above.

## Related
- Workspace root: `../CLAUDE.md`.
- ADRs explaining the layering: `../docs/adr/`.
