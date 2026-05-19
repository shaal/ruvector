# neural-trader/strategies

Strategy backtesting and ready-to-run strategy combinations.

## Files
- `backtesting.js` - Historical simulation with realistic slippage, using `@neural-trader/strategies` + `@neural-trader/backtesting` and RuVector pattern matching.
- `example-strategies.js` - Combined production strategies built on `../system/trading-pipeline.js` and `../system/backtesting.js`.

## Run
```
npm run strategies:backtest
node strategies/example-strategies.js
```

## Related
- Parent: `../CLAUDE.md`.
- Building blocks: `../system/`, `../production/`.
