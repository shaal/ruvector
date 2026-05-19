# edge-net/sim/tests

Integration and scenario tests for the simulator (CommonJS, `node:test`-based).

## Important files
- `integration.test.cjs` — broad end-to-end integration runs.
- `edge-cases.test.cjs` — pathological / boundary cases.
- `learning-lifecycle.test.cjs` — learning-phase lifecycle.
- `rac-coherence.test.cjs` — RAC coherence checks against the simulated network.

## Run
- `node --test tests/` from `../`. Or use `../test-quick.sh`.
