Jest/TypeScript test suite for `agentic-jujutsu` (quantum-resistant version-control system for AI agents). Standalone npm workspace - not part of the Rust workspace.

Files:
- `package.json` - private npm package, `jest` + `ts-jest`. Scripts: `test`, `test:integration`, `test:performance`, `test:validation`, `test:all`, `test:coverage`, `test:verbose`, `test:watch`.
- `jest.config.js` - ts-jest preset, matches `*.test.ts` and `*-tests.ts`, 30s timeout, 50% workers, 75/80/80/80 coverage thresholds.
- `integration-tests.ts` - cross-component integration coverage.
- `performance-tests.ts` - latency/throughput regression tests.
- `validation-tests.ts` - correctness / property-based validation.
- `run-all-tests.sh` - shell entry that runs all three suites; supports `--coverage` and `--verbose`.
- `TEST_RESULTS.md` - last-recorded run results / notes.

Install with `npm install`, then `npm test` or `./run-all-tests.sh`. Targets the agentic-jujutsu crates under `../../crates/agentic-jujutsu*/` (no Rust deps - exercises the JS/TS surface).
