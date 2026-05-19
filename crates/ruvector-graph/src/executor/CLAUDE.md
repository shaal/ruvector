# ruvector-graph/src/executor

High-performance query execution engine. Targets 100K+ traversals/second per core with sub-millisecond simple lookups and SIMD-optimized predicate evaluation.

## Files

- `mod.rs` — Module declarations.
- `plan.rs` — Logical and physical query plans.
- `operators.rs` — Vectorized operators (scan, filter, join, aggregate).
- `pipeline.rs` — Iterator-model pipeline execution.
- `parallel.rs` — Rayon-based parallel execution.
- `cache.rs` — Query result caching.
- `stats.rs` — Cost-based optimization statistics.

## Pointers

- Consumes plans built from `../cypher/` AST.
- SIMD primitives live in `../optimization/simd_traversal.rs`.
