# ruvix-vecgraph/src

## Files

- `lib.rs` — crate root; re-exports `KernelVectorStore`, `VectorStoreBuilder`, graph store types.
- `vector_store.rs` — `KernelVectorStore`: slab-region-backed vector store with capability + proof gating.
- `graph_store.rs` — slab-allocated graph store with proof-gated mutations.
- `hnsw.rs` — HNSW index built on slab-allocated nodes (fixed-size slots, zero allocator overhead).
- `simd_distance.rs` — SIMD distance kernels (per-arch).
- `coherence.rs` — coherence metadata co-located with each vector.
- `proof_policy.rs` — `ProofPolicy::standard()` etc. configuring which mutations require which proof tier.
- `witness.rs` — witness attestations emitted on every successful mutation.
