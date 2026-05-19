# ruvector-postgres/docker/test-runner

Test-runner container.

## Files

- `Dockerfile` — Image bundling the extension + test harness.
- `run-tests.sh` — Entrypoint script running `cargo pgrx test` + SQL tests.
