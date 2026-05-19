# mcp-brain-server

Cloud Run backend for the RuVector "Shared Brain". Provides a REST API (axum) for storing, searching, voting, and managing shared knowledge — every piece of knowledge is an RVF cognitive container with witness chains, Ed25519 signatures, and differential privacy proofs. Talks to Firestore + GCS as its persistence layer.

## Important files

- `Cargo.toml` — `mcp-brain-server` binary (axum REST), plus extra bins `ruvbrain-sse`, `ruvbrain-worker`, `mcp-brain-server-local`. Wires the entire RuVector cognitive stack (sona, ruvector-mincut, nervous-system, consciousness, solver, sparsifier, delta-core, domain-expansion, ruvllm, all `rvf/*` crates, midstream platform crates).
- `src/main.rs` — Cloud Run entrypoint: binds `0.0.0.0:$PORT`, starts background cognitive loop (5-min enhanced cycle + 60s tick).
- `src/lib.rs` — Re-exports all internal modules.
- `src/routes.rs` — axum router and HTTP handler implementations (large).
- `src/store.rs`, `src/gcs.rs`, `src/web_store.rs`, `src/web_memory.rs`, `src/web_ingest.rs` — storage / persistence (Firestore + GCS + Common Crawl ingest).
- `src/auth.rs`, `src/rate_limit.rs`, `src/reputation.rs`, `src/verify.rs` — auth, abuse control, signature verification.
- `src/cognitive.rs`, `src/symbolic.rs`, `src/voice.rs`, `src/optimizer.rs`, `src/trainer.rs`, `src/pipeline.rs`, `src/aggregate.rs`, `src/drift.rs` — cognitive loop (SONA + symbolic reasoning + internal voice + curiosity + GWT + LoRA federation).
- `src/embeddings.rs`, `src/quantization.rs`, `src/ranking.rs`, `src/graph.rs`, `src/gist.rs` — vector search / ranking (inlined 4x-unrolled cosine instead of ruvector-core SIMD).
- `src/midstream.rs` — real-time streaming analysis (nanosecond-scheduler, temporal-attractor-studio, strange-loop, optional temporal-neural-solver on x86).
- `src/notify.rs`, `src/pubmed.rs` — outbound integrations.
- `src/types.rs`, `src/tests.rs` — shared types and inline tests.
- `src/bin/ruvbrain_sse.rs` — Server-Sent Events streaming brain.
- `src/bin/ruvbrain_worker.rs` — background worker binary.
- `src/bin/local.rs` — local-mode binary (SQLite backend, `local` feature).
- `Dockerfile`, `Dockerfile.minimal`, `Dockerfile.sse`, `Dockerfile.trainer`, `Dockerfile.worker` — multi-image deployment.
- `cloudbuild*.yaml`, `deploy.sh`, `cloud/` — GCP Cloud Build / Cloud Run deploy configs.
- `scripts/`, `static/` — operational scripts and public web assets (agent guide, manifest, SEO files).

## Features

- `default = []`
- `x86-simd` — enables `temporal-neural-solver` (x86_64 only).
- `local` — enables SQLite backend (`rusqlite`).
- `local-all` — local + x86-simd.

## Related

- `crates/sona`, `crates/ruvector-mincut`, `crates/ruvector-nervous-system`, `crates/ruvector-consciousness`, `crates/ruvector-solver`, `crates/ruvector-sparsifier`, `crates/ruvector-domain-expansion`, `crates/ruvector-delta-core`, `crates/ruvllm`, `crates/rvf/*`.
- Midstream crates pulled from crates.io: `nanosecond-scheduler`, `temporal-attractor-studio`, `strange-loop`, `temporal-neural-solver`.
