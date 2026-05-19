# neural-trader/tests

Test suite and production benchmark runner.

## Files
- `neural-trader.test.js` - Jest tests covering the production modules in `../production/` and `../system/`.
- `production-benchmark.js` - CLI benchmark script for Fractional Kelly, Hybrid LSTM-Transformer, DRL Portfolio Manager, etc. Writes results consumed by `../docs/production-benchmark-results.md`.

## Run
```
npm test
node tests/production-benchmark.js
```

## Related
- Parent: `../CLAUDE.md`.
- Code under test: `../production/`, `../system/`.
