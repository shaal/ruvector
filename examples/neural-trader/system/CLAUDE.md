# neural-trader/system

Reusable building blocks combined by the CLI and the strategies/production examples.

## Files
- `backtesting.js` - Historical simulation framework (Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, Profit Factor, VaR, Expected Shortfall).
- `data-connectors.js` - Market data APIs (Yahoo, Alpha Vantage, Binance, Polygon.io).
- `risk-management.js` - Position limits, stop-loss (fixed/trailing/volatility-based), circuit breakers, exposure management.
- `trading-pipeline.js` - DAG-based pipeline wiring LSTM-Transformer prediction, sentiment alpha, DRL ensemble, and fractional Kelly sizing.
- `visualization.js` - ASCII terminal charts for equity curves, signals, metrics.

## Used by
- `../cli.js`, `../strategies/example-strategies.js`, `../full-integration/platform.js`.

## Related
- Parent: `../CLAUDE.md`.
