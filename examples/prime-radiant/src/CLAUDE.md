# prime-radiant/src

Library implementation for `prime-radiant-category`.

## Top-level files

- `lib.rs` - Public surface, re-exports, doc examples.
- `belief.rs` - Belief topos modeling uncertain knowledge.
- `coherence.rs` - Coherence verification utilities.
- `error.rs` - Error enum.
- `functor.rs` - Top-level functor traits (e.g. `EmbeddingFunctor`).
- `higher.rs` - Higher categories / coherence between morphisms.
- `natural_transformation.rs` - Natural transformations.
- `retrieval.rs` - Functorial retrieval system.
- `topos.rs` - Top-level topos definitions.

## Submodules

- `category/` - Core category theory primitives.
- `causal/` - Causal models, do-calculus, counterfactuals, abstraction.
- `cohomology/` - Chain complexes, homology, presheaves, sheaves.
- `hott/` - Homotopy Type Theory (types, paths, transport, equivalence, universes, checker).
- `quantum/` - Quantum states, density matrices, channels, topological codes/invariants, persistent homology.
- `spectral/` - Spectral analysis (Lanczos, Cheeger, clustering, energy, collapse).

## Related

- Tests: `../tests/`.
- Benches: `../benches/`.
- WASM bindings: `../wasm/`.
