# dna (rvDNA)

AI-native genomic analysis toolkit. Implements 20-SNP biomarker risk scoring, streaming anomaly detection, 64-dim DNA profile vectors, 23andMe genotyping import, CYP2D6/CYP2C19 pharmacogenomics, variant calling, protein structure prediction, and HNSW vector search in pure Rust.

## Important files
- `Cargo.toml` — published crate `rvdna` v0.3.0; bin `rvdna-cli`; three Criterion benches.
- `src/main.rs` / `src/lib.rs` — CLI + library entry.
- `src/{biomarker,biomarker_stream,genotyping,pharma,variant,protein,epigenomics,health}.rs` — domain modules.
- `src/{kmer,kmer_pagerank,alignment,rvdna,pipeline,real_data,types,error}.rs` — primitives, formats, pipelines.
- `benches/{dna_bench,solver_bench,biomarker_bench}.rs` — Criterion benchmarks.
- `tests/` — unit/integration tests (biomarker, kmer, pipeline, security).
- `adr/` — 15 Architecture Decision Records covering the design.
- `ddd/` — Domain-Driven Design documents (bounded contexts, architecture).

## Run
- CLI: `cargo run --release --bin rvdna-cli -- <subcommand>`.
- Benches: `cargo bench`.
- Tests: `cargo test`.

## Tech stack
- ruvector crates: `core`, `attention`, `gnn`, `graph`, `dag`, `math`, `filter`, `collections`, `solver` (forward-push/neumann/cg).
- `ndarray`, `tokio`, `tracing`, `uuid`, `chrono`.

## Related
- See ADR-013 for the rvDNA file format; ADR-015 for npm/WASM biomarker engine; sibling AI examples under `../`.
