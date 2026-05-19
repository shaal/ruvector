# dna/benches

Criterion benchmarks for rvDNA. Each file is a `[[bench]]` target in `../Cargo.toml` and produces HTML reports under `target/criterion/`.

## Important files
- `dna_bench.rs` — core DNA pipeline throughput.
- `solver_bench.rs` — sublinear PageRank / solver paths on k-mer graphs.
- `biomarker_bench.rs` — biomarker risk-scoring perf.

## Run
- `cargo bench` from `../` (the crate root). Use `cargo bench --bench dna_bench` to run one.

## Related
- See ADR-011 in `../adr/` for performance targets.
