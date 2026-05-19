# edge-net/sim

TypeScript lifecycle simulator for the edge-net P2P network. Spins up virtual contributors/consumers, drives them through learning, economic, and RAC phases, and emits a report.

## Important files
- `package.json` — scripts `build`, `simulate`, `simulate:fast`, `simulate:verbose`, `clean`.
- `tsconfig.json` — TS config.
- `src/` — TypeScript sources (cell, metrics, network, phases, report, simulator, plus some `.js` mirrors).
- `dist/` — compiled JS + sourcemaps (generated).
- `examples/quick-demo.js` — minimal usage demo.
- `scripts/generate-report.js`, `scripts/visualize.js` — post-run analysis.
- `tests/` — node-test integration / edge-case / lifecycle / RAC tests.
- `test-quick.sh` — quick smoke harness.
- `SIMULATION_*.md`, `USAGE.md`, `INDEX.md`, `PROJECT_SUMMARY.md`, `COMPLETION_REPORT.md` — docs.

## Run
- `npm install`
- `npm run simulate` (or `simulate:fast` / `simulate:verbose`).
- `npm run build` produces `dist/`.

## Tech stack
- Pure TS + Node 20, `ts-node`, `uuid`. No external runtime deps beyond uuid.
