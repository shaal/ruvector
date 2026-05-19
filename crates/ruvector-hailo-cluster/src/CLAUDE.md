# ruvector-hailo-cluster/src

Library + binaries for the multi-Pi Hailo embedding cluster. Library lints: `#![warn(missing_docs)]` (locked in iter 75 to prevent doc rot).

## Library files

- `lib.rs` — crate root, declares all modules, exposes `HailoClusterEmbedder`, `WorkerEndpoint`, `GrpcTransport`, the `transport::EmbeddingTransport` trait, plus public access to all submodules.
- `transport.rs` — `EmbeddingTransport` trait (object-safe; gRPC and fake-worker impls).
- `grpc_transport.rs` — `GrpcTransport` (tonic-based; optional rustls TLS via `tls` feature).
- `proto.rs` — generated tonic types re-exported.
- `pool.rs` — per-worker connection pool and lifecycle.
- `shard.rs` — request → worker routing (P2C + EWMA).
- `health.rs` — per-worker health tracking.
- `cache.rs` — optional in-process embedding cache (`with_cache(n)`).
- `discovery.rs` — static + Tailscale-tag worker discovery.
- `fingerprint.rs` — boot-time fingerprint enforcement; ejects mismatched workers.
- `manifest_sig.rs` — optional Ed25519 detached signature verification on `--workers-file` manifests (ADR-172 §1c, iter 107).
- `rate_limit.rs` — per-peer leaky-bucket rate limiter via `governor` + `dashmap` (ADR-172 §3b).
- `tls.rs` — TLS plumbing for both client and worker.
- `error.rs` — `thiserror` error type for the cluster.
- `bin/` — CLI binaries; see `bin/CLAUDE.md`.

## Public surface (re-exported from `lib.rs`)

`HailoClusterEmbedder`, `WorkerEndpoint`, `GrpcTransport`. All other modules are `pub mod` so binaries and downstream tests can reach the internals.

See `../CLAUDE.md`.
