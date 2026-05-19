# examples/vibecast-7sense

7sense - a domain-driven, multi-crate Rust workspace for the 7sense bioacoustics analysis platform: audio ingestion, Perch 2.0 ONNX embeddings, HNSW vector search, GNN-based learning, HDBSCAN clustering, LLM-powered interpretation, plus REST/GraphQL/WebSocket APIs. Built as a reference for layered (hexagonal) architecture on top of the RuVector stack.

## Top-level files
- `Cargo.toml` - Workspace manifest listing the nine `sevensense-*` crates plus integration `tests/`. Workspace deps include `tokio`, `axum`, `tracing-opentelemetry`, `symphonia`, `ort` (ONNX Runtime), `qdrant-client`, `ndarray`, `criterion`.
- `Cargo.lock`, `LICENSE` (MIT OR Apache-2.0).

## Subdirectories
- `crates/` - The nine domain crates (`sevensense-core/audio/embedding/vector/learning/analysis/interpretation/api/benches`).
- `assets/` - Architecture image.
- `benches/` - Workspace-level criterion benchmarks (api, clustering, embedding, hnsw, utils).
- `docs/` - DDD implementation plan, ADRs, and research plans.
- `scripts/` - `run_benchmarks.sh` plus a `performance_report.rs` reporter.
- `tests/` - Cross-crate integration tests + fixtures + mocks.

## Build
```
cargo build --workspace --release
cargo test --workspace
cargo bench -p sevensense-benches
```

## Tech stack
- Rust 2021 (rust-version 1.75), Tokio, Axum, ONNX Runtime (`ort` 2.0-rc), Qdrant client, OpenTelemetry, Symphonia, Criterion.

## Related
- RuVector crates: `crates/ruvector-*` in the repo root provide lower-level primitives that some 7sense crates re-use.
- For trading-domain DDD see `examples/neural-trader/system/`.
