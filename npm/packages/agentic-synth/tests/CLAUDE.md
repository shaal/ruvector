# agentic-synth/tests

Test suites for `@ruvector/agentic-synth`, run via Vitest (`npm run test`, `test:unit`, `test:integration`, `test:cli`).

## Top-level files

- `dspy-learning-session.test.ts` — end-to-end DSPy.ts learning session test.
- `manual-install-test.js` — manual install smoke test.

## Subdirectories

- `unit/` — fast isolated tests (api, cache, config, generators, routing).
- `integration/` — integration tests against `ruvector`, `agentic-robotics`, `midstreamer` adapters.
- `cli/` — CLI behavior tests (commander parsing, exit codes).
- `fixtures/` — shared test fixtures (`configs.js`, `schemas.js`).
- `training/` — DSPy training tests.
