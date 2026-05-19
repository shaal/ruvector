# ruvector-postgres/src/graph/cypher

Simplified Cypher query support inside the PostgreSQL extension (subset of Cypher; for full Cypher see sibling crate `ruvector-graph`).

## Files

- `mod.rs` — Re-exports `ast::*`, `executor::execute_cypher`, `parser::parse_cypher`.
- `parser.rs` — `parse_cypher` lexing + parsing.
- `ast.rs` — AST types.
- `executor.rs` — `execute_cypher` evaluator against `graph::storage` tables.
