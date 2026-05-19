# dna/src

Source tree for the `rvdna` library and `rvdna-cli` binary.

## Important files
- `lib.rs` — public crate API.
- `main.rs` — `rvdna-cli` entry point.
- `types.rs` / `error.rs` — common types and errors.
- `pipeline.rs` — orchestrates analysis pipelines (DAG-driven via `ruvector-dag`).
- `kmer.rs` / `kmer_pagerank.rs` — k-mer graphs and sublinear PageRank.
- `alignment.rs` — sequence alignment.
- `genotyping.rs` — 23andMe-format genotype ingestion.
- `variant.rs` — variant calling.
- `biomarker.rs` / `biomarker_stream.rs` — 20-SNP risk scoring (streaming + batch).
- `pharma.rs` — CYP2D6/CYP2C19 pharmacogenomics.
- `protein.rs` — protein structure prediction (GNN-backed).
- `epigenomics.rs` — epigenomic time series modeling.
- `health.rs` — health-biomarker analysis.
- `real_data.rs` — real-dataset loaders.
- `rvdna.rs` — native rvDNA file format (ADR-013).

## Build
- From parent: `cargo build --release`.
