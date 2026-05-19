# neural-trader/core

Core integration patterns between neural-trader packages and RuVector.

## Files
- `basic-integration.js` - Initialize neural-trader with RuVector backend, run basic trading ops, compare native-Rust vs JS performance.
- `hnsw-vector-search.js` - Native HNSW indexing via NAPI (claimed 150x faster than pure JS).
- `technical-indicators.js` - 150+ technical indicators via `@neural-trader/features` with RuVector-backed caching / pattern matching.

## Run
```
npm run core:basic
npm run core:hnsw
npm run core:features
```

## Related
- Parent: `../CLAUDE.md`.
- Used by strategies/backtesting: `../strategies/`, `../system/`.
