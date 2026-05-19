# exo-ai-2025/tests

Workspace-level integration tests for the EXO-AI substrate. 28 tests
total across substrate / hypergraph / temporal / federation surfaces
plus three cross-crate scenarios; written TDD-style per
`../INTEGRATION_TESTS_COMPLETE.md`.

## Files

- `substrate_integration.rs` — core substrate workflow (5 tests).
- `hypergraph_integration.rs` — hypergraph topology + sheaf surface (6
  tests).
- `temporal_integration.rs` — STM/LTM, causal links, decay (8 tests).
- `federation_integration.rs` — CRDT, consensus, handshake, onion (9
  tests).
- `full_stack_test.rs` — substrate + manifold + temporal + federation
  end-to-end.
- `manifold_hypergraph_test.rs` — manifold over a hypergraph.
- `temporal_federation_test.rs` — replicated temporal memory.

## Subdirectory

- `common/` — shared test helpers (fixtures, assertions, utilities).

## Run

```bash
cargo test --workspace
# or via the runner:
bash ../scripts/run-integration-tests.sh
```

## Related

- `../docs/INTEGRATION_TEST_GUIDE.md`, `../docs/TEST_INVENTORY.md`
- `../test-templates/` — earlier TDD scaffolds
