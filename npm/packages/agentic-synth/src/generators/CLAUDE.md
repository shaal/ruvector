# agentic-synth/src/generators

Concrete data generators used by `@ruvector/agentic-synth`. Also exposed as subpath export `@ruvector/agentic-synth/generators` via tsup.

## Files

- `base.ts` — abstract base class with shared retry, caching, and provider-call logic.
- `timeseries.ts` — `TimeSeriesGenerator` for time-indexed numeric series.
- `events.ts` — `EventGenerator` for discrete event streams.
- `structured.ts` — `StructuredGenerator` for arbitrary JSON shaped by a user schema.
- `data-generator.js` — unit-test target for the base generator.
- `index.ts` — barrel re-export of the above.

Each `.ts` file has compiled `.js` and `.d.ts` siblings.
