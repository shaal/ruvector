# ruvector-verified

Formal-verification layer for RuVector using `lean-agentic` dependent types. Provides proof-carrying vector operations, verified pipeline composition, and formal attestation for safety-critical paths — with sub-microsecond proof overhead under the `ultra` feature.

## Important files

- `Cargo.toml` — depends on `lean-agentic` (workspace) and `thiserror`. Optional integrations: `ruvector-core` (with `hnsw` feature) for `hnsw-proofs`, `ruvector-coherence` for `coherence-proofs`, `ruvector-cognitive-container` for `rvf-proofs`. Optional `serde` for proof serialization. Performance features: `fast-arena`, `simd-hash`, `gated-proofs`, aggregated as `ultra`. `all-proofs` = `hnsw-proofs + rvf-proofs + coherence-proofs`.
- `src/lib.rs` — module declarations and feature gating.

## Module map (src/)

Always compiled:
- `invariants.rs` — runtime / compile-time invariant types (e.g. dimension, normalization).
- `vector_types.rs` — proof-tagged vector types.
- `pipeline.rs` — verified pipeline composition.
- `proof_store.rs` — store / lookup of generated proofs.
- `pools.rs` — object pools.
- `cache.rs` — proof / hash cache.
- `error.rs` — crate error enum.

Feature-gated:
- `fast_arena.rs` (`fast-arena`) — SolverArena-style bump allocator.
- `gated.rs` (`gated-proofs`) — coherence-gated proof-depth routing.

## Benches

- `benches/arena_throughput.rs` — `fast-arena` throughput.
- `benches/proof_generation.rs` — proof-generation latency (sub-µs target).

## Public API surface

Verified vector types, pipeline composition primitives, proof-store API, invariant types, and the integration adapters enabled by feature flags.

## Related

- `crates/lean-agentic` (workspace dep) — dependent-types backbone.
- `crates/ruvector-core` (`hnsw-proofs`), `crates/ruvector-coherence` (`coherence-proofs`), `crates/ruvector-cognitive-container` (`rvf-proofs`).
- `crates/ruvector-graph-transformer` consumes this with `features=["ultra","hnsw-proofs"]`.
