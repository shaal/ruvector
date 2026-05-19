# ruvector-verified/src

Formal-verification source. Some files are feature-gated.

## Always compiled

- `lib.rs` — module declarations.
- `error.rs` — crate error enum.
- `invariants.rs` — runtime / compile-time invariant types.
- `vector_types.rs` — proof-tagged vector types using `lean-agentic` dependent types.
- `pipeline.rs` — verified pipeline composition.
- `proof_store.rs` — proof generation, storage, lookup.
- `pools.rs` — object pools for proof artifacts.
- `cache.rs` — proof / hash-cons cache.

## Feature-gated

- `fast_arena.rs` (`fast-arena`) — SolverArena-style bump allocator for sub-µs proof paths.
- `gated.rs` (`gated-proofs`) — coherence-gated proof-depth routing.

SIMD hashing (`simd-hash` feature) toggles AVX2/NEON paths inside `cache.rs` and `proof_store.rs`.
