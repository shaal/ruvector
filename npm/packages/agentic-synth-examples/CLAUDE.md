# @ruvector/agentic-synth-examples

Production-ready examples for `@ruvector/agentic-synth`: DSPy multi-model training, benchmarking, self-learning, stock-market, security, CI/CD, and multi-agent swarm data generators. Distributed as both an installable library and a CLI.

## Key files

- `package.json` — `@ruvector/agentic-synth-examples` v0.1.0; ESM-only (`"type": "module"`), built with `tsup`.
- `tsconfig.json`, `tsup.config.ts`, `vitest.config.ts` — TS/build/test config.
- `CHANGELOG.md` — release notes.
- `bin/cli.js` — `agentic-synth-examples` CLI entry.
- `src/index.ts` — main barrel: re-exports DSPy classes, `SelfLearningGenerator`, `StockMarketSimulator`, `SecurityTestingGenerator`, `CICDDataGenerator`, `SwarmCoordinator`, and an `Examples` factory.

## Subdirectories

- `bin/` — CLI launcher.
- `docs/` — `QUICK-START-TESTING.md`, `TEST-SUITE-SUMMARY.md`.
- `examples/` — narrative example scripts (`beginner/`, `intermediate/`, `advanced/`).
- `src/` — TypeScript source (with prebuilt `.js`/`.d.ts`); one folder per domain (`dspy`, `generators`, `cicd`, `security`, `self-learning`, `stock-market`, `swarm`, `types`).
- `tests/` — `vitest` integration + per-domain unit tests.

## Published API

- `.` -> all generators + factory; types for metrics/configs.
- `./dspy` -> DSPy training session, benchmark, model agents (Claude/GPT4/Llama/Gemini).

## Scripts

- `build` / `build:dspy` / `build:all` -> `tsup` ESM+CJS+dts
- `dev` -> watch mode
- `test` / `test:watch` / `test:coverage` / `test:ui` -> vitest
- `typecheck` -> `tsc --noEmit`

## Key deps

`@ruvector/agentic-synth` (peer), `dspy.ts`, `commander`, `zod`.

## Related

- Sibling: `npm/packages/agentic-synth/` (the library these examples consume).
