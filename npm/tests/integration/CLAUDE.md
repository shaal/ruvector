# npm/tests/integration

Cross-package integration tests that load the top-level `ruvector` meta-package and verify the native (`@ruvector/core`) and WASM (`@ruvector/wasm`) backends expose compatible behaviour.

## Files

- `cross-package.test.js` - Uses `node:test`. Asserts that `ruvector.getBackendInfo()` reports a valid backend (`native` or `wasm`), that `new ruvector.VectorIndex({ dimension })` works against the loaded backend, and that the API surface is identical across backends. Falls back to WASM when no native binding is available.

## Running

From `npm/tests`: `node --test integration/` or `node run-all-tests.js --only=integration`.

## Related

- `../unit/` - Per-backend unit tests.
- `../../packages/ruvector/` - The meta-package whose backend-selection logic these tests verify.
