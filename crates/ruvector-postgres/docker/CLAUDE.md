# ruvector-postgres/docker

Docker assets for development, testing, integration testing, and benchmarking the PostgreSQL extension.

## Files

- `Dockerfile` — Main development image.
- `Dockerfile.test` — Test-runner image.
- `Dockerfile.integration-test` — Integration-test image.
- `docker-compose.yml` — Local stack.
- `docker-compose.integration.yml` — Integration-test stack.
- `init.sql`, `init-integration.sql` — Database init scripts.
- `postgresql.conf` — Tuned config for benchmarking.
- `dev.sh` — Local dev orchestrator.
- `run-tests.sh` — Run the unit/test image.
- `run-integration-tests.sh` — Run integration-test stack.
- `publish-dockerhub.sh` — Push images to Docker Hub.

## Subdirectories

- `baseline/` — Baseline image (currently empty / placeholder).
- `benchmark/` — Benchmark-runner image + script.
- `test-runner/` — Test-runner image + script.
