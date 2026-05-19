# test/

Integration tests and benchmarks for `ruvector`. Run via `npm test` (which executes `integration.js` then `cli-commands.js`).

- `integration.js` — end-to-end integration smoke test (entry for `npm test`).
- `cli-commands.js` — exercises the `ruvector` CLI commands.
- `standalone-test.js` — standalone harness independent of the CLI.
- `mock-implementation.js` — mock backend used to test fallback paths.
- `optimizer.test.js` — unit tests for `src/optimizer/`.
- `decompiler-reconstruction.js` — coverage for the `src/decompiler/` pipeline.

## Benchmarks

- `benchmark-cli.js`, `benchmark-gnn.js`, `benchmark-hooks.js`, `benchmark-perf.js` — perf measurements across CLI, GNN wrapper, hooks, and core ops.
