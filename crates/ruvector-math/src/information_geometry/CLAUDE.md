# ruvector-math/src/information_geometry

Information geometry: treats probability distributions as points on a curved manifold, enabling geometry-aware optimisation.

- `mod.rs` — re-exports `FisherInformation`, `NaturalGradient`, `KFACApproximation`.
- `fisher.rs` — `FisherInformation` (Fisher information matrix).
- `natural_gradient.rs` — `NaturalGradient` descent that respects manifold geometry.
- `kfac.rs` — `KFACApproximation` (Kronecker-factored approximate curvature).

References: Amari & Nagaoka (2000); Martens & Grosse (2015); Pascanu & Bengio (2013).

See `../CLAUDE.md`.
