# ruvix/benches/src

Shared library code used by both the criterion harnesses (in `../benches/`) and the binary entry points (in `bin/`).

## Files

- `lib.rs` — crate root re-exporting the helper modules below.
- `comparison.rs` — utilities to run a measurement against both RuVix and Linux backends and produce a paired result.
- `linux.rs` — Linux syscall wrappers used as the baseline.
- `ruvix.rs` — RuVix syscall wrappers used as the system under test.
- `report.rs` — formatted reporting (tables, markdown, csv).
- `stats.rs` — statistical aggregation (mean/median/percentile helpers).
- `targets.rs` — definitions of the benchmark targets / scenarios.
- `bin/` — see `bin/CLAUDE.md`.
