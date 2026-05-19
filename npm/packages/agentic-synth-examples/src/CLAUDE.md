# src/

TypeScript source for `@ruvector/agentic-synth-examples`. Compiled with `tsup` into `dist/` (ESM+CJS+dts) — but `.js`/`.d.ts` copies are also checked into this tree.

- `index.ts` — main barrel; re-exports DSPy classes, all five generators (self-learning, stock-market, security, CICD, swarm), plus an `Examples` factory object.

Subdirectories (each typically with an `index.ts`):

- `dspy/` — DSPy training session, multi-model benchmark, per-provider model agents.
- `generators/` — `SelfLearningGenerator`, `StockMarketSimulator`.
- `cicd/` — `CICDDataGenerator` for pipeline/test/deployment/monitoring data.
- `security/` — `SecurityTestingGenerator` for vuln/anomaly/pentest scenarios.
- `self-learning/` — adaptive feedback loop generator.
- `stock-market/` — OHLCV + news event simulator.
- `swarm/` — `SwarmCoordinator` for multi-agent coordination.
- `types/` — shared type definitions barrel.
