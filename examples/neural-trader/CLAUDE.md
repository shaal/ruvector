# examples/neural-trader

Comprehensive Node.js example suite integrating the `neural-trader` packages with the RuVector vector database. Demonstrates 20+ `@neural-trader/*` packages across core integration, strategies, portfolio, neural networks, risk, MCP, accounting, specialized markets, and exotic ML.

## Top-level files
- `package.json` - ES module, Node >=18. Workspace of >20 `@neural-trader/*` deps plus `@ruvector/core`. Defines an `npm run`-able script per example (e.g. `core:basic`, `exotic:rl`, `full:platform`).
- `cli.js` - Standalone CLI (`npx neural-trader run|backtest|paper`) that wires `system/trading-pipeline.js`, `system/backtesting.js`, `system/data-connectors.js`, `system/risk-management.js`.

## Subdirectories
- `accounting/` - Crypto tax / cost-basis demos using `@neural-trader/agentic-accounting-rust-core`.
- `advanced/` - Conformal prediction, live Alpaca broker, order-book microstructure.
- `core/` - Basic integration, HNSW vector search, technical indicators.
- `docs/` - Production benchmark results.
- `exotic/` - GNN, attention, RL, quantum, hyperbolic, swarm, atomic arbitrage; includes a benchmark.
- `full-integration/` - End-to-end platform example wiring all packages.
- `mcp/` - MCP server exposing 87+ trading tools.
- `neural/` - Neural network training (LSTM etc.).
- `portfolio/` - Markowitz, risk parity, max Sharpe, min vol.
- `production/` - DRL portfolio, fractional Kelly, hybrid LSTM-Transformer, sentiment alpha.
- `risk/` - VaR, CVaR, drawdown, Sharpe/Sortino/Calmar.
- `specialized/` - News trading, prediction markets, sports betting.
- `strategies/` - Backtesting + ready-to-run example strategies.
- `system/` - Reusable building blocks: trading pipeline (DAG), backtesting, data connectors, risk management, visualization.
- `tests/` - Jest test suite and production benchmark script.

## Run
```
npm install
npm run core:basic
npm run full:platform
node cli.js run --strategy=hybrid --symbol=AAPL
npm test
```

## Tech stack
- Node 18+, ES modules, Jest.
- `@neural-trader/*` packages, `@ruvector/core`.

## Related
- Apify Actors wrapping neural-trader: `examples/apify/`.
- Cloud-hosted benchmarks: `examples/google-cloud/`.
