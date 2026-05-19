# test-templates/integration

Workspace-level TDD scaffolding for cross-crate integration paths.

## Files

- `full_stack_test.rs` — exercises substrate + manifold + temporal +
  federation in one scenario.
- `manifold_hypergraph_test.rs` — combined manifold + hypergraph
  pipeline.
- `temporal_federation_test.rs` — temporal memory replicated across a
  federation.

Live versions of these live alongside per-crate tests in `../../tests/`.

## Related

- `../../tests/` — live workspace integration tests
- `../../INTEGRATION_TESTS_COMPLETE.md` — TDD plan
