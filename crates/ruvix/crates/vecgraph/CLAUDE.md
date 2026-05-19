# ruvix-vecgraph

Kernel-resident vector and graph stores for the RuVix Cognition Kernel (ADR-087 Section 4.3). Unlike conventional kernels,
vector/graph data structures here are first-class kernel objects: data lives in kernel-managed slab regions with capability
protection, HNSW index nodes are slab-allocated, coherence metadata is co-located with each vector, all mutations are proof-gated,
and every successful mutation emits a witness attestation.

## Syscalls implemented

- `vector_get` — read vector + coherence metadata (capability-gated, no proof).
- `vector_put_proved` — write vector with proof verification (PROVE right).
- `graph_apply_proved` — apply graph mutation with proof verification.

## Files

- `Cargo.toml` — depends on `ruvix-types` + `ruvix-region`. Dev: criterion, proptest.
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.
- `tests/proof_gated.rs` — proof-gated mutation tests.
