# ruvector-math

Advanced mathematics library for next-generation vector search and AI governance. Pure-Rust (no BLAS/LAPACK), WASM-compatible, SIMD-friendly hot paths. Modules are designed around the principle of "mincut as the unifying spine" — every subsystem can integrate with the mincut governance signal.

## Module groups

- **Optimal transport**: Wasserstein, Sinkhorn, Sliced Wasserstein, Gromov-Wasserstein.
- **Information geometry**: Fisher information, natural gradient, K-FAC.
- **Product manifolds**: Mixed-curvature spaces (Euclidean × Hyperbolic × Spherical).
- **Spherical geometry**: Geodesics on the n-sphere for cyclical patterns.
- **Tropical algebra**: Max-plus semiring for piecewise-linear analysis and routing.
- **Tensor networks**: TT / Tucker / CP decomposition for memory compression.
- **Spectral methods**: Chebyshev polynomials for graph diffusion without eigendecomposition.
- **Persistent homology** (`homology`): TDA for topological drift detection.
- **Polynomial optimisation** (`optimization`): SOS certificates for provable bounds.
- **Utilities** (`utils`) and an `error` module.

## Features

- `default = ["std"]`
- `simd` — opt-in SIMD kernels.
- `parallel = ["rayon"]`
- `serde = ["dep:serde"]`

## Layout

- `Cargo.toml` — `nalgebra` (no default features), `rand`, `rand_distr`, `thiserror`; optional `rayon`, `serde`. Dev: `criterion`, `proptest`, `approx`.
- `src/` — module tree; see `src/CLAUDE.md`.
- `benches/` — criterion suites for `optimal_transport`, `information_geometry`, `product_manifold`, `spectral`, `tropical`.

## Public API

Top-level re-exports per module; e.g. `ruvector_math::optimal_transport::{SlicedWasserstein, SinkhornSolver, OptimalTransport}`, `ruvector_math::information_geometry::{FisherInformation, NaturalGradient, KFACApproximation}`, etc.

## Related

- `crates/ruqu-core` — quantum simulation that consumes some of these primitives (tensor networks, spectral).
- `crates/ruvector-dag/src/attention/hierarchical_lorentz.rs` — uses hyperbolic / product-manifold utilities.
- `crates/cognitum-gate-kernel` and `crates/neural-trader-coherence` — consume mincut signals built on this foundation.
