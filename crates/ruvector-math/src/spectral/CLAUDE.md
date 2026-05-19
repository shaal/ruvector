# ruvector-math/src/spectral

Spectral methods over graphs without eigendecomposition. Uses Chebyshev polynomials for O(K·m) graph filters.

- `mod.rs` — re-exports.
- `chebyshev.rs` — Chebyshev polynomial recurrence and graph filtering.
- `graph_filter.rs` — high-level filter API.
- `clustering.rs` — spectral clustering / k-way partitioning.
- `wavelets.rs` — multi-scale spectral wavelet transform.

Pairs naturally with mincut: mincut finds partition boundaries, Chebyshev smooths within them. See `../CLAUDE.md`.
