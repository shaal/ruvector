# npm/tests/unit

Per-package unit tests using `node:test` and `node:assert`. Each file exercises one published surface in isolation; tests that need a native binding or built WASM module gracefully skip when those artifacts are missing.

## Files

- `core.test.js` - Tests `@ruvector/core`: platform detection, `VectorDB` construction (with and without full `HnswConfig`), `insert` / `insertBatch` / `search` / `delete` / `get` / `len` / `isEmpty`, sort order of search results, and `version` / `hello` utility functions.
- `wasm.test.js` - Tests `@ruvector/wasm` via dynamic `import()`: module loading, environment detection, async `init()`, basic CRUD, and search.
- `ruvector.test.js` - Tests the top-level `ruvector` meta-package: backend detection (`getBackendInfo`, `isNativeAvailable`), `VectorIndex`, exposed `Utils` namespace.
- `cli.test.js` - Spawns `../../ruvector/bin/ruvector.js` via `child_process` to verify the CLI's commands, error handling, and output formatting. Uses `../fixtures/temp/` for scratch state.

## Running

From the `npm/tests` directory: `node --test unit/` or via the driver `node run-all-tests.js --only=unit`.

## Related

- `../integration/cross-package.test.js` - Verifies these backends interoperate.
- `../run-all-tests.js` - The aggregate driver.
