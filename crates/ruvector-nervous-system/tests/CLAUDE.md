# ruvector-nervous-system/tests

Integration tests (`cargo test --test <name>`).

## Files

- `integration.rs` — broad smoke test.
- `btsp_integration.rs` — BTSP plasticity end-to-end.
- `eprop_tests.rs` — e-prop online learning.
- `ewc_tests.rs` — Elastic Weight Consolidation.
- `workspace_integration.rs` — Global Workspace routing.
- `memory_bounds.rs` — memory-usage bound tests.
- `retrieval_quality.rs` — Hopfield / HDC retrieval-quality checks.
- `throughput.rs` — throughput assertions (gated by feature for CI stability).

## Subdirectories

- `integration/` — additional integration test files (see its CLAUDE.md).
