# refrag-pipeline

Demo of the REFRAG (Compress-Sense-Expand) pipeline for ~30x latency reduction in RAG systems. Builds a vector store, runs synthetic queries, and reports timings vs. baselines.

## Important files
- `Cargo.toml` — bins `refrag-demo` (`src/main.rs`) and `refrag-benchmark` (`src/benchmark.rs`); Criterion bench `refrag_bench`.
- `src/lib.rs` — public REFRAG API.
- `src/types.rs` — `RefragEntry`, `RefragResponseType`.
- `src/compress.rs` — `CompressionStrategy` (compress step).
- `src/sense.rs` — `PolicyNetwork` (sense step, chooses retrieval policy).
- `src/expand.rs` — `ExpandLayer` (expand step).
- `src/store.rs` — `RefragStoreBuilder` (vector store).
- `src/main.rs` — demo runner (search_dim=384, tensor_dim=768, 1000 docs / 100 queries).
- `src/benchmark.rs` — full benchmark binary.
- `benches/refrag_bench.rs` — Criterion microbenchmarks.

## Run
- Demo: `cargo run --release --bin refrag-demo`.
- Benchmark: `cargo run --release --bin refrag-benchmark`.
- Microbench: `cargo bench`.

## Tech stack
- `../../crates/ruvector-core`, `ndarray`, `tokio`, `tracing`, `uuid`, `chrono`, `base64`.
