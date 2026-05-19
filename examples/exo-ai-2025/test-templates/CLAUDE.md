# exo-ai-2025/test-templates

Earlier TDD scaffolding: per-crate test directories laid out the same
way as the real crates, plus a shared `integration/` directory. These
templates seeded the tests that now live under
`../crates/<crate>/tests/` and `../tests/`.

## Subdirectories

- `exo-backend-classical/tests/classical_backend_test.rs`
- `exo-core/tests/core_traits_test.rs`
- `exo-federation/tests/federation_test.rs`
- `exo-hypergraph/tests/hypergraph_test.rs`
- `exo-manifold/tests/manifold_engine_test.rs`
- `exo-temporal/tests/temporal_memory_test.rs`
- `integration/` — three workspace-level integration tests
  (`full_stack_test.rs`, `manifold_hypergraph_test.rs`,
  `temporal_federation_test.rs`) mirroring `../tests/`.

These files are kept as a snapshot of the initial TDD design and may
drift slightly from the live tests.

## Related

- `../crates/<crate>/tests/` — live unit/integration tests
- `../tests/` — live workspace integration tests
