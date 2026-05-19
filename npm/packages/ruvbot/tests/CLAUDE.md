# ruvbot / tests

Vitest test tree for ruvbot. Configured by `../vitest.config.ts`.

## Files
- `setup.ts` - Global vitest setup (env, mocks).
- `index.ts` - Aggregates shared test exports (factories, fixtures,
  mocks).

## Subdirectories
- `unit/` - Fast unit tests per module (api, core, domain, plugins,
  security, wasm, workers).
- `integration/` - Cross-module tests (postgres, slack, ruvector wasm,
  multitenancy, swarm, hybrid search).
- `e2e/` - End-to-end conversation / skill / long-running task flows.
- `factories/` - Test data factories.
- `fixtures/` - Static fixtures.
- `mocks/` - Postgres / Slack mocks.
