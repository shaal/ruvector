# exo-backend-classical/tests

Integration tests for the classical backend.

## Files

- `classical_backend_test.rs` — end-to-end backend correctness on the
  exo-core trait set.
- `learning_benchmarks.rs` — learning-rule perf checks.
- `performance_comparison.rs` — head-to-head vs. baseline references.
- `transfer_pipeline_test.rs` — exercises `transfer_orchestrator` and
  the `ruvector-domain-expansion` bridge.

## Run

```bash
cargo test -p exo-backend-classical
```

## Related

- `../src/` — module-under-test
- `../../../test-templates/exo-backend-classical/` — earlier TDD scaffolds
