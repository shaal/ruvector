# @ruvector/agentic-synth

High-performance synthetic data generator for AI/ML training, RAG systems, and agentic workflows. Wraps DSPy.ts, Gemini, and OpenRouter to produce structured/time-series/event data, with optional integration into the `ruvector` vector DB and `agentic-robotics`/`midstreamer` peers.

## Layout

- `package.json` — `@ruvector/agentic-synth` v0.1.6. Main `dist/index.cjs`, ESM `dist/index.js`, types `dist/index.d.ts`. Subpath exports for `./generators` and `./cache`. Bin: `agentic-synth -> ./bin/cli.js`. Deps: `@google/generative-ai`, `commander`, `dotenv`, `dspy.ts`, `zod`. Peer deps: `ruvector`, `agentic-robotics`, `midstreamer` (optional).
- `src/index.ts` — entry point exporting the `AgenticSynth` class with `generate`, `generateTimeSeries`, `generateEvents`, `generateStructured` methods. Initializes generators and parses config via Zod.
- `src/types.ts` — Zod schemas + TS types for `SynthConfig`, `ModelProvider`, `DataType`, `GenerationResult`, etc.
- `src/generators/` — generator implementations (timeseries, events, structured, base).
- `src/cache/` — `context-cache.ts` and cache index (memory/disk caching).
- `src/api/`, `src/routing/`, `src/adapters/`, `src/config/` — API client, model router (Gemini/OpenRouter), adapters for ruvector/robotics/midstreamer, and config loader.
- `bin/cli.js` — Commander-based CLI entry.
- `config/synth.config.example.json` — sample user config.
- `benchmark.js`, `test-example.js`, `test-live-api.js` — top-level scripts.
- `tests/`, `examples/`, `training/`, `docs/` — see subdir CLAUDE.md files.

## Scripts

- `build` / `build:all` (tsup ESM+CJS+dts), `dev` (watch), `test` / `test:unit` / `test:integration` / `test:cli` (vitest), `typecheck`, `lint`, `format`, `benchmark`.

## Related

- Optional peer: `ruvector` (top-level npm package), `npm/packages/ruvector-extensions`.
- Examples live under `examples/<domain>/` (ad-roas, crypto, security, swarms, etc).
