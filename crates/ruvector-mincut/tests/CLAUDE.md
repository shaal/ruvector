# ruvector-mincut/tests

Integration tests (`cargo test --test <name>`).

## Files

- `integration_tests.rs` — end-to-end exercise of `MinCutBuilder` / `DynamicMinCut`.
- `bounded_integration.rs` — bounded-instance variant.
- `canonical_bench.rs` — correctness tests for the canonical decomposition.
- `certificate_tests.rs` — proof-certificate generation and verification.
- `jtree_tests.rs` — J-tree hierarchy correctness.
- `localkcut_integration.rs` — deterministic local k-cut integration tests.
- `localkcut_paper_integration.rs` — paper-faithful local k-cut tests.
- `paper_algorithm_tests.rs` — tests for the published paper algorithms.
- `wrapper_tests.rs` — `MinCutWrapper` API tests.
- `coverage_tests.rs` — line/branch-coverage-driven tests.
