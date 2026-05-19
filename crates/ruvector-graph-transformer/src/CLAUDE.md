# ruvector-graph-transformer/src

Implementation of the proof-gated graph transformer. Each module is feature-gated (see `Cargo.toml`).

## Files

- `lib.rs` — module declarations and feature map.
- `proof_gated.rs` — the core proof-gated mutation substrate (always compiled).
- `config.rs` — shared configuration types.
- `error.rs` — crate error enum.

## Feature-gated modules

- `sublinear_attention.rs` (`sublinear`) — LSH + PPR-sampled sublinear attention.
- `verified_training.rs` (`verified-training`) — per-step proof certificates during training.
- `physics.rs` (`physics`) — Hamiltonian graph networks.
- `biological.rs` (`biological`) — spiking + STDP / Hebbian.
- `self_organizing.rs` (`self-organizing`) — morphogenetic / L-system growth.
- `manifold.rs` (`manifold`) — product-manifold attention.
- `temporal.rs` (`temporal`) — causal / Granger attention.
- `economic.rs` (`economic`) — Shapley + game-theoretic attention.
