# cognitum-gate-tilezero/tests_disabled

Parked tests that are not included in `cargo test` (directory name has
underscore so Cargo does not auto-discover). Keep for reference; re-enable by
moving back into `tests/` (likely behind the `audit-replay` feature).

## Files
- `replay_tests.rs` - replay verification tests targeting `src/replay.rs`.
  Disabled pending stabilization of the replay API.
