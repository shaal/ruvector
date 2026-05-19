# npm/tests

Cross-package test suite for the npm side of the monorepo. Exercises `@ruvector/core` (native), `@ruvector/wasm`, the top-level `ruvector` meta-package, and the `ruvector` CLI - then a set of integration tests verifies backends are interchangeable, plus optional performance benchmarks. Tests use the built-in `node:test` runner; no Jest / Vitest dependency.

## Files

- `run-all-tests.js` - Driver that spawns `node --test` for each test file under `unit/`, `integration/`, and optionally `performance/`. Supports `--perf` and `--only=<category>` flags. Emits `test-results.json` with a pass/fail summary.
- `QUICK_START.md`, `TEST_RESULTS.md`, `TEST_SUMMARY.md` - Human-readable docs for running tests and tracking results.

## Subdirectories

- `unit/` - Per-package unit tests (`core.test.js`, `wasm.test.js`, `ruvector.test.js`, `cli.test.js`).
- `integration/` - Cross-package tests (`cross-package.test.js`) verifying backend-agnostic APIs.
- `performance/` - Optional benchmarks (`benchmarks.test.js`), skipped by default.

## Running

- `node run-all-tests.js` - Run unit + integration suites.
- `node run-all-tests.js --perf` - Include performance benchmarks.
- `node run-all-tests.js --only=unit` (or `integration`, `performance`).

## Notes

- Tests gracefully skip when native bindings or built WASM aren't present (looking for `MODULE_NOT_FOUND` / `ERR_MODULE_NOT_FOUND`).
- The CLI test resolves `../../ruvector/bin/ruvector.js` and writes scratch data into `fixtures/temp/` (created on demand).

## Related

- `../core/`, `../wasm/` - The two backend packages under test.
- `../packages/ruvector/` - The meta-package exposing `VectorIndex`, `getBackendInfo`, `isNativeAvailable`, `Utils`.
- `../packages/ruvector-cli/` (or similar) - CLI binary exercised by `unit/cli.test.js`.
