# cognitum-gate-kernel/tests_disabled

Tests held outside the active `tests/` directory so they do not run with `cargo test`. Kept here so a future agent can re-enable them as the kernel API stabilises.

- `evidence_tests.rs` — `EvidenceAccumulator` properties.
- `integration.rs` — end-to-end tile lifecycle.
- `report_tests.rs` — tick `Report` shape/contents.
- `shard_tests.rs` — `CompactGraph` invariants.

To re-enable a file, move it into `../tests/`. See parent `../CLAUDE.md`.
