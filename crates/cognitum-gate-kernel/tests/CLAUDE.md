# cognitum-gate-kernel/tests

Active integration tests for the tile kernel.

- `security_tests.rs` — exercises the threat model from `../SECURITY.md` / `../docs/SECURITY_AUDIT.md`.
- `canonical_witness_bench.rs` — correctness + perf checks for canonical-witness emission (feature-gated).

For quarantined tests not yet wired into CI, see `../tests_disabled/CLAUDE.md`.
