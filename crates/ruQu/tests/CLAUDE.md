# ruQu/tests

Integration tests for the `ruqu` crate.

## Files

- `filter_tests.rs` — Signal-filter correctness.
- `integration_tests.rs` — End-to-end pipeline (syndrome ingest -> gate decision).
- `stress_tests.rs` — High-load stress under many concurrent syndromes/tiles.
- `syndrome_tests.rs` — `DetectorBitmap` and `SyndromeRound` semantics.
- `tile_tests.rs` — 256-tile fabric behavior.

Run via `cargo test -p ruqu`.
