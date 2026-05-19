# ruvector-graph/fuzz/fuzz_targets

Individual `cargo-fuzz` targets.

## Files

- `fuzz_cypher_parser.rs` — Feeds arbitrary bytes into `parse_cypher` to check for panics/UB in the parser/lexer.

Run via `cargo +nightly fuzz run fuzz_cypher_parser`.
