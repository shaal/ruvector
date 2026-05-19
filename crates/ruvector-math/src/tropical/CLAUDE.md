# ruvector-math/src/tropical

Tropical algebra (max-plus semiring). Replaces (×, +) with (max, +) or (min, +). Used to analyse ReLU neural networks as tropical rational functions, to plan shortest paths / routing, and to formalise dynamic programming.

- `mod.rs` — re-exports `TropicalMatrix`, `MinPlusMatrix`, `TropicalEigen`, `TropicalNeuralAnalysis`, `LinearRegionCounter`.
- `semiring.rs` — semiring traits (max-plus / min-plus).
- `matrix.rs` — `TropicalMatrix`, `MinPlusMatrix`, eigenvalues (`TropicalEigen`).
- `polynomial.rs` — tropical polynomials (piecewise linear).
- `neural_analysis.rs` — `TropicalNeuralAnalysis`, `LinearRegionCounter` (bounds on linear regions of ReLU nets).

See `../CLAUDE.md`.
