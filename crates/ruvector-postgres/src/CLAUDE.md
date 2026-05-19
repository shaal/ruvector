# ruvector-postgres/src

Source root for the PostgreSQL extension. `lib.rs` declares all modules via `::pgrx::pg_module_magic!()` and registers GUCs.

## Top-level files

- `lib.rs` — `pg_module_magic!`, module declarations, GUC registration. Allows many WIP lints.
- `operators.rs` — Top-level operator definitions (cross-cutting).
- `bin/pgrx_embed.rs` — pgrx embedded build helper binary.

## Submodules (each has its own CLAUDE.md)

- `attention/` — Attention operators (Flash, multi-head, scaled-dot, MoE, hyperbolic, sparse, GAT).
- `dag/` — Neural DAG learning for query optimization; submodule `functions/` for SQL functions, `state`, GUC, hooks, worker.
- `distance/` — SIMD distance functions (scalar + simd).
- `domain_expansion/` — Cross-domain transfer learning.
- `embeddings/` — Local embedding generation via fastembed-rs.
- `gated_transformer/` — Mincut-gated transformer (feature-gated).
- `gnn/` — GCN, GraphSAGE, message passing, aggregators.
- `graph/` — Graph storage + traversal + Cypher + SPARQL.
- `healing/` — Self-healing engine.
- `hybrid/` — Hybrid BM25+vector search.
- `hyperbolic/` — Poincaré + Lorentz models.
- `index/` — `ruhnsw` + `ruivfflat` access methods.
- `integrity/` — Stoer-Wagner mincut gating.
- `learning/` — Trajectory tracking + ReasoningBank + pattern extraction.
- `math/` — Spectral methods + distances (via `ruvector-math`).
- `quantization/` — Scalar/Product/Binary.
- `routing/` — Tiny Dancer FastGRNN routing.
- `solver/` — Linear solvers (via `ruvector-solver`).
- `sona/` — Sona self-learning (per-table+dim cached engines).
- `sparse/` — Sparse vectors (COO).
- `tda/` — Topological data analysis.
- `tenancy/` — Multi-tenancy: isolation, quotas, RLS, registry, validation.
- `types/` — Vector types (`RuVector` f32, `HalfVec` f16, `SparseVec`, `BinaryVec`, `ProductVec`, `ScalarVec`).
- `workers/` — Background workers.

## Pointers

- SQL extension scripts: `../sql/ruvector--*.sql`.
- Tests: `../tests/*.sql`.
- Benchmarks: `../benches/*.rs`.
