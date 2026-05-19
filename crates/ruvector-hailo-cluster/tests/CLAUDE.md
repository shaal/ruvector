# ruvector-hailo-cluster/tests

Integration tests covering the library, the CLI binaries, and end-to-end transport / security paths.

## Library + load behaviour

- `cluster_load_distribution.rs` — P2C + EWMA dispatch fairness across simulated workers.
- `dos_gates.rs` — DoS / abuse gates (rate limiter, manifest signature).
- `rate_limit_interceptor.rs` — per-peer leaky bucket via `governor` (ADR-172 §3b).
- `secure_stack_composition.rs` — joint TLS + rate-limit + fingerprint composition.

## TLS

- `tls_roundtrip.rs` — server TLS roundtrip.
- `mtls_roundtrip.rs` — mutual TLS roundtrip.
- `mmwave_bridge_tls.rs` — TLS on the mmWave bridge path.

## CLI smoke tests

- `embed_cli.rs`, `bench_cli.rs`, `stats_cli.rs`, `mmwave_bridge_cli.rs`, `ruview_csi_bridge_cli.rs`, `ruvllm_bridge_cli.rs`.

## Hardware

- `pi_hardware_integration.rs` — gated tests that talk to a real Pi 5 + Hailo-8.

## Shared helpers

- `common/` — see `common/CLAUDE.md`.

See `../CLAUDE.md`.
