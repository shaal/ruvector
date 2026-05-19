# ruvector-postgres

High-performance PostgreSQL extension (`pgrx` cdylib) — a pgvector drop-in replacement with 230+ SQL functions, SIMD acceleration (AVX-512/AVX2/NEON), Flash Attention, GNN layers, hybrid (BM25+vector) search, multi-tenancy, self-healing, and self-learning. Provides the `ruhnsw` and `ruivfflat` index access methods.

## Important files

- `Cargo.toml` — `crate-type = ["cdylib", "lib"]`. Features per Postgres version: `pg14`/`pg15`/`pg16`/`pg17` (default `pg17`). SIMD features (native/avx2/avx512/neon/auto). Quantization features (scalar/product/binary/all). Index features (`index-hnsw`, `index-ivfflat`). Optional `embeddings`, `gated-transformer`. `pg_test` for `pgrx-tests`.
- `Cargo.lock` — Lockfile.
- `Dockerfile`, `Dockerfile.prebuilt` — Container images for building/running.
- `Makefile`, `DOCKERHUB.md`, `LEARNING_MODULE_COMPLETE.txt`, `GRAPH_MODULE_DELIVERY.md`, `SPARSE_DELIVERY.md` — Build/delivery notes.
- `build.rs` — Build script (pgrx integration).
- `ruvector.control` — PostgreSQL extension control file.
- `src/lib.rs` — `pgrx::pg_module_magic!()`, declares all modules, GUC registration.

## Source layout (`src/`) — each submodule has its own CLAUDE.md

- `attention/` — 39 attention mechanisms (Flash, MoE, hyperbolic, sparse, GAT/v2).
- `dag/` — Neural DAG learning for query optimization (SONA integration); contains `functions/`, `state`, GUC, hooks, worker.
- `distance/` — SIMD-optimized distance functions (AVX-512/AVX2/NEON + scalar fallback).
- `domain_expansion/` — Cross-domain transfer learning.
- `embeddings/` — Local embedding generation via fastembed-rs (ONNX models).
- `gated_transformer/` — Mincut-gated transformer (feature-gated).
- `gnn/` — Graph neural networks: GCN, GraphSAGE, message passing, aggregators.
- `graph/` — Graph storage + traversal + Cypher + SPARQL.
- `healing/` — Self-healing engine: detector, strategies, learning, worker.
- `hybrid/` — BM25 + vector hybrid search with RRF/linear/learned fusion.
- `hyperbolic/` — Poincaré ball + Lorentz hyperboloid embeddings.
- `index/` — HNSW + IVFFlat access methods (`ruhnsw`, `ruivfflat`).
- `integrity/` — Stoer-Wagner mincut gating + integrity contracts.
- `learning/` — Self-learning trajectory + ReasoningBank + pattern extraction.
- `math/` — Distances + spectral methods (wraps `ruvector-math`).
- `quantization/` — Scalar/Product/Binary quantization.
- `routing/` — Tiny Dancer FastGRNN neural routing.
- `solver/` — Linear solvers exposed as SQL functions.
- `sona/` — SONA self-learning engine (per-table+dimension cached).
- `sparse/` — Sparse vectors (COO format) for BM25/SPLADE.
- `tda/` — Topological data analysis (persistent homology).
- `tenancy/` — Multi-tenancy: isolation levels, quotas, RLS, registry.
- `types/` — Vector types: `RuVector` (f32), `HalfVec` (f16), `SparseVec`, `BinaryVec`, `ProductVec`, `ScalarVec`.
- `workers/` — Background workers (engine, GNN, integrity, maintenance, queue).
- `bin/pgrx_embed.rs` — pgrx embedded build helper.
- `operators.rs` — Top-level operator wiring.

## Other directories

- `benches/` — Criterion benchmarks (distance, e2e, hybrid, index, integrity, quantization).
- `docker/` — Compose files + Dockerfiles for tests and benchmarks.
- `docs/` — Extensive architecture, API, build, security, and integration plans.
- `examples/` — SQL/markdown usage examples.
- `install/` — Cross-platform installer (`install.sh`, `quick-start.sh`, `scripts/setup-*.sh`, config + verifier).
- `scripts/` — `docker-test.sh`, `download_models.rs` (model fetch for embeddings).
- `sql/` — Extension SQL files: `ruvector--0.1.0.sql`, `ruvector--0.3.0.sql`, `ruvector--2.0.0.sql`, `ruvector--2.0.0--0.3.0.sql`, plus per-feature examples.
- `tests/` — SQL integration tests.

## Build / Run

```
cargo pgrx install --features pg17
```

## Related

- Wraps many ruvector crates: `ruvector-math`, `ruvector-solver`, `ruvector-sona`, `ruvector-domain-expansion`, `ruvector-attention`, `ruvector-tiny-dancer-core`, `ruvector-hyperbolic-hnsw`, etc.
- Pgvector compatibility surface — drop-in replacement.
