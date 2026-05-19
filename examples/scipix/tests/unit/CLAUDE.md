# scipix/tests/unit

Unit tests for individual scipix modules.

## Files

- `mod.rs` - Module surface.
- `config_tests.rs` - `config.rs` tests.
- `error_tests.rs` - `error.rs` tests.
- `math_tests.rs` (~19 KB) - Math AST / parser / formatters.
- `ocr_tests.rs` - OCR engine internals.
- `output_tests.rs` - Output formatters.
- `preprocess_tests.rs` - Preprocessing pipeline.

## How to run

```bash
cargo test -p ruvector-scipix --lib
```

## Related

- Production code: `../../src/`.
- Integration tests: `../integration/`.
