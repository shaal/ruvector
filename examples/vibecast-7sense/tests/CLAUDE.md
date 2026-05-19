# vibecast-7sense/tests

Cross-crate integration tests for the 7sense workspace.

## Files
- `Cargo.toml` - Integration-test crate that depends on every `sevensense-*` crate.
- `lib.rs` - Crate root pulling in fixtures, mocks, and integration tests.

## Subdirectories
- `integration/` - Per-bounded-context integration tests (analysis, api, audio, embedding, interpretation, vector).
- `fixtures/` - Shared test fixtures (sample audio metadata, deterministic embeddings).
- `mocks/` - Mock implementations of repository traits used by tests.

## Run
```
cargo test -p tests
```

## Related
- Crates under test: `../crates/`.
