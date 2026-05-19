# agentic-synth/src

TypeScript source for `@ruvector/agentic-synth`. Compiled to `dist/` via tsup.

## Files

- `index.ts` — main entry. Exports the `AgenticSynth` class with `generate`, `generateTimeSeries`, `generateEvents`, `generateStructured` methods. Initializes the three generators with a Zod-validated `SynthConfig`.
- `types.ts` — Zod schemas + TS types: `SynthConfig`, `SynthConfigSchema`, `ModelProvider`, `DataType`, `GenerationResult`, `GeneratorOptions`, `TimeSeriesOptions`, `EventOptions`, etc.

## Subdirectories

- `generators/` — concrete generators (timeseries, events, structured, base).
- `cache/` — `context-cache` and cache index (memory/disk strategies).
- `api/` — provider HTTP client (`client.js`).
- `routing/` — `model-router.js` deciding Gemini vs OpenRouter.
- `adapters/` — peer integrations: `ruvector.js`, `robotics.js`, `midstreamer.js`.
- `config/` — config loader (`config.js`).
- `types/` — additional type re-exports.

Note: many compiled `.js` files exist beside `.ts` source.
