# agentic-synth/src/cache

Caching layer for `@ruvector/agentic-synth` generations.

## Files

- `index.ts` — exports the `CacheManager` class used by generators. Selects strategy (`none`, `memory`, `disk`) from `SynthConfig.cacheStrategy`.
- `context-cache.js` — context cache implementation (Gemini context-caching style).

Also published as a subpath export `@ruvector/agentic-synth/cache` via tsup.
