# edge-net/sim/src

TypeScript source for the lifecycle simulator.

## Important files
- `simulator.ts` — top-level entry; orchestrates the run.
- `cell.ts` — virtual cell (participant) model.
- `network.ts` / `network.js` — virtual P2P network.
- `phases.ts` / `phases.js` — lifecycle phase definitions.
- `metrics.ts` — metrics collection.
- `report.ts` — final report generation.
- `economics.js`, `node.js` — additional simulator pieces (JS-only).

## Build
- From `../`: `npm run build` (emits `../dist/`).
