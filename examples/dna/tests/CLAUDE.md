# dna/tests

Integration tests for the rvDNA crate.

## Important files
- `biomarker_tests.rs` — 20-SNP biomarker scoring correctness.
- `kmer_tests.rs` — k-mer graph and PageRank.
- `pipeline_tests.rs` — end-to-end pipeline runs.
- `security_tests.rs` — privacy / security invariants (per ADR-012).

## Run
- `cargo test` from `../` (the crate root).
