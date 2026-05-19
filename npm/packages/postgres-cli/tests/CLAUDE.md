# postgres-cli/tests

Install / smoke tests for `@ruvector/postgres-cli`, primarily exercising the `npx` install path under a clean Docker image.

## Files

- `Dockerfile.npx-test` — minimal image used to verify `npx @ruvector/postgres-cli` works on a fresh system.
- `test-npx-install.sh` — runner script for the Dockerized install test.
