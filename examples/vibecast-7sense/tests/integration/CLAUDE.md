# vibecast-7sense/tests/integration

Integration tests exercising each bounded context end-to-end.

## Files
- `mod.rs` - Aggregates per-context test modules.
- `analysis_test.rs` - Clustering / motif / sequence flows (`sevensense-analysis`).
- `api_test.rs` - HTTP / GraphQL / WebSocket flows (`sevensense-api`).
- `audio_test.rs` - Audio ingestion + segmentation (`sevensense-audio`).
- `embedding_test.rs` - Perch 2.0 inference (`sevensense-embedding`).
- `interpretation_test.rs` - Report / insight generation (`sevensense-interpretation`).
- `vector_test.rs` - HNSW indexing / search (`sevensense-vector`).

## Run
```
cargo test -p tests
```

## Related
- Fixtures: `../fixtures/`.
- Mocks: `../mocks/`.
