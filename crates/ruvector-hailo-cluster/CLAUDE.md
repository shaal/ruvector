# ruvector-hailo-cluster

Multi-Pi cluster coordinator for ruvector's Hailo-8 embedding workers (ADR-167 §8 "hailo-backend"). Distributes embed requests across a fleet of Pi 5 + Hailo-8 nodes with P2C+EWMA load balancing, fingerprint enforcement, optional in-process caching, and Tailscale-tag-based discovery. Also hosts host-side bridge binaries for mmWave radar, RuView CSI, and the in-tree ruvLLM engine.

Internal-only (`publish = false`).

## Features

- `grpc` (reserved; gRPC compiles unconditionally).
- `hailo` — propagates `ruvector-hailo/hailo` so the worker binary actually talks to `/dev/hailo0`. Off by default so x86 dev hosts can still build.
- `tls` — `tonic/tls` (rustls). Defence-in-depth; Tailscale already encrypts the wire.
- `cpu-fallback` — propagates `ruvector-hailo/cpu-fallback` for host-CPU BERT-6 fallback.
- `ruvllm-engine` — wires in-tree `ruvllm` engine into `ruvllm-pi-worker`.

## Layout

- `Cargo.toml` — see features above; many tonic / prost / tokio / tracing deps. Six `[[bin]]` targets (see `src/bin/`).
- `build.rs` — invokes `tonic-build` on `proto/embedding.proto`.
- `deny.toml`, `.cargo/` — cargo-deny and cargo config.
- `BENCHMARK.md`, `RUVLLM_CLUSTER_PLAN.md`, `RUVLLM_NEXT_PLAN.md` — operator docs.
- `src/` — library + bins; see `src/CLAUDE.md`.
- `benches/dispatch.rs` — load-balancer dispatch benchmark.
- `examples/hailo-cluster-as-provider.rs` — example showing `HailoClusterEmbedder` implementing `ruvector_core::EmbeddingProvider`.
- `proto/embedding.proto` — gRPC service definition.
- `deploy/` — systemd unit files, install scripts, udev rules, HEF compilation tools (see `deploy/CLAUDE.md`).
- `tests/` — integration tests for CLI bins, mTLS, DoS gates, rate-limits, hardware (see `tests/CLAUDE.md`).

## Binaries (in `src/bin/`)

- `ruvector-hailo-worker` — Hailo-side gRPC server (consumes `/dev/hailo0` with feature `hailo`).
- `ruvector-hailo-embed` — client: stdin / `--text` → JSONL embeddings.
- `ruvector-hailo-fakeworker` — host-side mock for tests.
- `ruvector-hailo-stats` — fleet observability (TSV / JSON / Prom).
- `ruvector-hailo-cluster-bench` — sustained-load harness.
- `ruvector-mmwave-bridge` — feeds mmWave radar bytes (via `ruvector-mmwave`) to the cluster.
- Plus `ruview-csi-bridge.rs`, `ruvllm-bridge.rs`, `ruvllm-pi-worker.rs` (see `src/bin/CLAUDE.md`).

## Public library API

`HailoClusterEmbedder::new(workers, transport, dim, fingerprint).with_cache(n)`, `validate_fleet()`, `embed_one_blocking(text)` / `embed_one(...)`; `WorkerEndpoint`, `GrpcTransport`, the `transport::EmbeddingTransport` trait, fingerprint enforcement, pool, rate-limit, TLS, discovery, cache, manifest signing.

## Related

- `crates/ruvector-hailo` — the per-node embedding worker library (driven by `ruvector-hailo-worker` here).
- `crates/ruvector-mmwave` — shared mmWave UART parser used by `ruvector-mmwave-bridge`.
- `crates/ruvector-core` — `EmbeddingProvider` trait that `HailoClusterEmbedder` implements (ADR-178 Gap B).
- `crates/ruvllm` — LLM engine used when `ruvllm-engine` feature is on.
