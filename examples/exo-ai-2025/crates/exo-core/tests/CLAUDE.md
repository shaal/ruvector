# exo-core/tests

Trait-contract tests for the exo-core public surface.

## Files

- `core_traits_test.rs` — exercises `Substrate`, `Witness`,
  `CoherenceRouter`, etc. against stub implementations to lock down the
  expected behavior of each trait method.

## Run

```bash
cargo test -p exo-core
```

## Related

- `../src/traits.rs` — contracts under test
- `../../../test-templates/exo-core/tests/` — earlier TDD scaffold
