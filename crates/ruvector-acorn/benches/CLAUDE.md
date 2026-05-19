# ruvector-acorn/benches

Criterion benchmarks for ACORN.

- `acorn_bench.rs` — registered as `[[bench]] name = "acorn_bench"`. Compares
  `FlatFilteredIndex`, `AcornIndex1`, and `AcornIndexGamma` on QPS / recall across
  selectivity levels.

Run: `cargo bench -p ruvector-acorn --bench acorn_bench`.
