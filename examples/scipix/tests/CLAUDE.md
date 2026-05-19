# scipix/tests

Test suite for scipix - both unit and integration tests, with shared fixtures and helpers.

## Files

- `lib.rs` - Test crate entry.
- `math_tests.rs` (~18 KB) - Math parser / formatter tests at the top level.
- `SUMMARY.md` - Test plan summary.

## Subdirs

- `common/` - Shared test helpers (images, latex, metrics, server, types).
- `unit/` - Unit tests (config, error, math, ocr, output, preprocess).
- `integration/` - End-to-end tests (accuracy, api, cache, cli, performance, pipeline).
- `fixtures/` - Static fixtures (configs, expected outputs).

## How to run

```bash
cargo test -p ruvector-scipix
cargo test -p ruvector-scipix --test math_tests
```

## Related

- Docs: `../docs/11_TEST_STRATEGY.md`.
