# ruvector-math/src/spherical

Operations on the n-sphere S^n. Used for cyclical patterns (time-of-day, seasonal), directional data, and normalised embeddings where cosine similarity is the natural metric.

- `mod.rs` — entire module in one file. Provides `SphericalConfig { max_iterations, threshold }` plus geodesic distance (`arccos(<x,y>)`), exponential / logarithmic maps, and Fréchet mean.

Internals use helpers from `../utils` (`dot`, `norm`, `normalize`, `EPS`). See `../CLAUDE.md`.
