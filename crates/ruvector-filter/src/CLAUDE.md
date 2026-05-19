# ruvector-filter/src

Source root for the metadata-filter crate.

## Files

- `lib.rs` — Crate doc, module declarations, public re-exports. Sets `recursion_limit = 4096` for deeply nested filter ASTs.
- `expression.rs` — `FilterExpression` enum + builder methods (eq, ne, gt/gte/lt/lte, in/not_in, geo, text, and/or/not).
- `index.rs` — `PayloadIndexManager` (DashMap-backed), `IndexType { Keyword, Integer, Float, Boolean, Geo, Text }`.
- `evaluator.rs` — Walks a `FilterExpression` against `PayloadIndexManager` to produce matching IDs.
- `error.rs` — Crate error enum + `Result`.
