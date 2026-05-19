# exo-ai-2025/tests/common

Shared helpers re-imported by every integration test in `../`.

## Files

- `mod.rs` — module roots (re-exports the helpers below).
- `fixtures.rs` — synthetic substrate / hypergraph / temporal /
  federation fixtures.
- `assertions.rs` — custom assertions tailored to substrate state and
  consciousness metrics.
- `helpers.rs` — utility functions (setup/teardown, deterministic RNG,
  timing).

## Related

- `../substrate_integration.rs`, `../hypergraph_integration.rs`,
  `../temporal_integration.rs`, `../federation_integration.rs`,
  `../full_stack_test.rs`, `../manifold_hypergraph_test.rs`,
  `../temporal_federation_test.rs`
