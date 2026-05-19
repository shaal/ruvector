# scipix/tests/integration

End-to-end integration tests for scipix.

## Files

- `mod.rs` - Module surface.
- `accuracy_tests.rs` - OCR accuracy vs. golden expected outputs.
- `pipeline_tests.rs` - Full preprocess -> OCR -> output pipeline.
- `api_tests.rs` - HTTP API endpoint tests (uses `../common/server.rs`).
- `cache_tests.rs` - Cache behaviour under load.
- `cli_tests.rs` - CLI invocation tests.
- `performance_tests.rs` - Performance/regression thresholds.

## How to run

```bash
cargo test -p ruvector-scipix --test '*'
```

## Related

- Shared helpers: `../common/`.
- Fixtures: `../fixtures/`.
