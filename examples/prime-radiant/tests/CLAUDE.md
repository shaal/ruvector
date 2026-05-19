# prime-radiant/tests

Integration tests for the prime-radiant modules. Note: `Cargo.toml` only enables `integration_tests`; the per-module test files exist but are gated off (modules need refinement before re-enabling).

## Files

- `integration_tests.rs` - Cross-module integration tests (enabled).
- `category_tests.rs`, `cohomology_tests.rs`, `hott_tests.rs`, `spectral_tests.rs`, `causal_tests.rs`, `quantum_tests.rs` - Per-module tests (currently commented out in manifest; kept for reference).

## How to run

```bash
cargo test -p prime-radiant-category --test integration_tests
```

## Related

- Implementations: `../src/`.
- Inline lib tests live alongside their modules.
