# ruvector-math/src/utils

Shared numerical and sorting helpers used throughout the crate.

- `mod.rs` — re-exports `numerical::*` and `sorting::*`; defines constants `EPS = 1e-10`, `EPS_F32 = 1e-7`, `LOG_MIN = -700.0`, `LOG_MAX = 700.0`.
- `numerical.rs` — `dot`, `norm`, `normalize`, and other numerically-stable kernels (log-sum-exp, clamping, etc.).
- `sorting.rs` — sorting helpers (e.g. partial sort for top-k).

See `../CLAUDE.md`.
