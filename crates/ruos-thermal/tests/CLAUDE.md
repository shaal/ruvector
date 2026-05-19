# ruos-thermal/tests

Integration tests for the `ruos-thermal` CLI/library.

## Files

- `cli.rs` — Drives the binary against synthetic sysfs paths (uses `tempfile` dev-dep) and asserts parsed snapshots.
