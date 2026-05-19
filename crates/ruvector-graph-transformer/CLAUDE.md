# ruvector-graph-transformer

Unified graph transformer with a proof-gated mutation substrate. Composes existing RuVector crates (`ruvector-verified`, `ruvector-gnn`, `ruvector-attention`, `ruvector-mincut`) into one façade with formal verification guarantees, providing 8 specialised graph-intelligence modules: physics, biological, manifold, temporal, economic, self-organizing, sublinear attention, and verified training.

## Important files

- `Cargo.toml` — feature-gated module surface (`sublinear`, `verified-training` default; plus `physics`, `biological`, `self-organizing`, `manifold`, `temporal`, `economic`; `full` enables all).
- `src/lib.rs` — module map and feature documentation.
- `tests/integration.rs` — integration test across enabled feature modules.

## Module map (src/)

- `proof_gated.rs` — core proof-gated mutation types (the substrate that wraps all other modules).
- `sublinear_attention.rs` — O(n log n) attention via LSH and Personalized PageRank sampling.
- `physics.rs` — Hamiltonian graph networks with energy-conservation proofs.
- `biological.rs` — spiking attention with STDP and Hebbian learning.
- `self_organizing.rs` — morphogenetic fields and L-system graph growth.
- `verified_training.rs` — GNN training with per-step proof certificates.
- `manifold.rs` — product manifold attention on S^n × H^m × R^k.
- `temporal.rs` — causal temporal attention with Granger causality.
- `economic.rs` — game-theoretic / Shapley / incentive-aligned attention.
- `config.rs` — shared configuration.
- `error.rs` — crate error enum.

## Related

- `crates/ruvector-verified` (ultra + hnsw-proofs), `crates/ruvector-gnn`, `crates/ruvector-attention`, `crates/ruvector-mincut` — upstream building blocks.
- `crates/ruvector-attn-mincut`, `crates/ruvector-mincut-gated-transformer` — adjacent specialised transformers.
