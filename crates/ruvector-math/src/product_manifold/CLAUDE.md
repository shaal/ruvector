# ruvector-math/src/product_manifold

Mixed-curvature product manifolds: M = H^h × E^e × S^s. Combine hyperbolic (hierarchy), Euclidean (general embeddings), and spherical (cyclical) components in one space.

- `mod.rs` — re-exports `ProductManifold`, `ProductManifoldConfig`, `CurvatureType`; also a `#[doc(hidden)] pub mod ops` for batched operations used internally.
- `config.rs` — `CurvatureType` enum, `ProductManifoldConfig`.
- `manifold.rs` — `ProductManifold` type and core operations (exp, log, parallel transport).
- `operations.rs` — batched / vectorised manifold operations.

Targets ~20x memory reduction on taxonomy data vs pure Euclidean. References: Gu et al. (2019), Skopek et al. (2020). See `../CLAUDE.md`.
