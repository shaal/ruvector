# ruvector-graph/src/cypher

Complete Cypher query language implementation: tokenization, parsing, semantic analysis, and query optimization. Supports hyperedges (N-ary relationships).

## Files

- `mod.rs` — Module declarations and re-exports (`Query`, `Statement`, `Token`, `TokenKind`, `OptimizationPlan`, `QueryOptimizer`, `parse_cypher`, `ParseError`, `SemanticAnalyzer`, `SemanticError`).
- `lexer.rs` — Tokenization (`Token`, `TokenKind`).
- `parser.rs` — Syntax parsing -> AST (`parse_cypher`, `ParseError`).
- `ast.rs` — AST types (`Query`, `Statement`, clauses).
- `semantic.rs` — Type checking + scope resolution (`SemanticAnalyzer`, `SemanticError`).
- `optimizer.rs` — Logical/physical query optimization (`QueryOptimizer`, `OptimizationPlan`).

## Pointers

- Execution lives in `../executor/`.
- Hybrid extensions (vector predicates) live in `../hybrid/cypher_extensions.rs`.
- Fuzzing harness: `../../fuzz/fuzz_targets/fuzz_cypher_parser.rs`.
