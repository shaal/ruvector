# prime-radiant/src/category

Core category theory abstractions: objects, morphisms, functors, natural transformations, and concrete instantiations (`SetCategory`, `VectorCategory`, topoi).

## Files

- `mod.rs` - Module surface, `Category` trait and laws verification.
- `object.rs` - Generic object trait.
- `morphism.rs` - Morphism trait, composition, identity.
- `functor.rs` - Functor trait between categories.
- `natural.rs` - Natural transformations between functors.
- `set_category.rs` (~19KB) - Concrete `SetCategory`.
- `vector_category.rs` (~22KB) - Vector space category for embeddings.
- `topos.rs` - Topos-theoretic structures.

## Related

- ADR: `../../docs/adr/ADR-002-category-topos.md`.
- Higher coherence: `../higher.rs`.
- Bench: `../../benches/category_bench.rs`.
