# mcp-brain-server/src

All server source. Each file is a top-level module re-exported from `lib.rs`.

## Entrypoints

- `main.rs` — Cloud Run binary (binds `PORT`, spawns cognitive loop).
- `lib.rs` — declares and re-exports all modules.
- `bin/` — extra binaries (SSE, worker, local SQLite).

## HTTP layer

- `routes.rs` — full axum router and handlers (largest file).
- `routes.rs.backup` — previous version kept for reference.
- `auth.rs` — request auth (Ed25519 / signed payloads).
- `rate_limit.rs` — per-key rate limiting via `dashmap`.
- `reputation.rs` — reputation scoring used by ranking and rate limiting.
- `verify.rs` — signature / witness chain verification.
- `notify.rs` — outbound notifications.

## Storage

- `store.rs` — primary store façade.
- `web_store.rs`, `web_memory.rs`, `web_ingest.rs` — web-side store (Firestore docs + GCS blobs).
- `gcs.rs` — GCS REST helpers (uses `reqwest`).

## Vector / search

- `embeddings.rs` — embedding generation (HashEmbedder, RlmEmbedder via `ruvllm`).
- `quantization.rs` — vector compression.
- `graph.rs` — DiskANN-style graph index; inline 4x-unrolled cosine (no SIMD intrinsics, for Docker portability).
- `ranking.rs`, `aggregate.rs`, `gist.rs` — search ranking and result aggregation.

## Cognition

- `cognitive.rs` — SONA + symbolic loop driver.
- `symbolic.rs` — symbolic reasoning.
- `voice.rs` — internal voice / curiosity.
- `optimizer.rs` — meta-optimization.
- `trainer.rs` — training loop (LoRA federation).
- `pipeline.rs` — staged pipeline composition.
- `drift.rs` — drift detection.
- `midstream.rs` — real-time stream analysis (nanosecond-scheduler, temporal-attractor-studio, strange-loop).
- `pubmed.rs` — PubMed scientific-paper ingest source.

## Shared

- `types.rs` (+ `types.rs.backup`) — shared API types.
- `tests.rs` — inline integration tests.
