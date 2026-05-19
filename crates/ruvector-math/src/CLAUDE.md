# ruvector-math/src

Module map.

- `lib.rs` — crate root, declares all modules and the architecture overview.
- `error.rs` — `MathError`, `Result` alias via `thiserror`.
- `utils/` — `EPS`, `EPS_F32`, `LOG_MIN/MAX`, numerical helpers (`dot`, `norm`, `normalize`, ...), sorting helpers.
- `optimal_transport/` — Wasserstein / Sinkhorn / Sliced-Wasserstein / Gromov-Wasserstein.
- `information_geometry/` — Fisher, natural gradient, K-FAC.
- `product_manifold/` — mixed-curvature Euclidean × Hyperbolic × Spherical manifolds.
- `spherical/` — operations on the n-sphere.
- `tropical/` — max-plus semiring, matrices, polynomials, neural-network linear-region analysis.
- `tensor_networks/` — TT / Tucker / CP decompositions.
- `spectral/` — Chebyshev graph filters, clustering, wavelets.
- `homology/` — persistent homology (filtration, persistence diagrams, distances, simplicial complexes).
- `optimization/` — SOS certificates, moment relaxations, polynomial / SDP solvers.

Each subdirectory has its own `CLAUDE.md`. See `../CLAUDE.md`.
