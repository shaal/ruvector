# ruvector-filter

Advanced metadata filtering for ruvector vector search. Provides flexible filter expressions (equality, range, geo, text, AND/OR/NOT), efficient payload indexing (integer, float, keyword, boolean, geo, text), and fast filter evaluation using indices.

## Important files

- `Cargo.toml` — Workspace member. Depends on `ruvector-core` (path), `serde`, `serde_json`, `thiserror`, `dashmap`, `uuid`, `chrono`, `ordered-float`.
- `src/lib.rs` — Crate root. `#![recursion_limit = "4096"]`. Doc-comment includes a runnable example combining `FilterExpression`, `PayloadIndexManager`, `FilterEvaluator`, `IndexType`.

## Source modules (`src/`)

- `expression.rs` — `FilterExpression` AST (eq/ne, gt/gte/lt/lte, in/not_in, geo, text, and/or/not).
- `index.rs` — `PayloadIndexManager` + `IndexType` enum (Keyword, Integer, Float, Boolean, Geo, Text).
- `evaluator.rs` — `FilterEvaluator` that executes expressions against the index manager.
- `error.rs` — Error type.

## Public API

- `FilterExpression` (and constructors: `eq`, `gte`, `and`, ...).
- `PayloadIndexManager` (create_index, index_payload).
- `FilterEvaluator::new(&manager).evaluate(&filter)`.
- `IndexType`.

## Related

- Backbone: `ruvector-core`.
- Used downstream by query layers in `ruvector-postgres` and `ruvector-graph`.
