# neural-trader/production

Production-grade trading components backed by recent research.

## Files
- `drl-portfolio-manager.js` - Ensemble of PPO, SAC, A2C for dynamic portfolio allocation.
- `fractional-kelly.js` - Fractional Kelly criterion for safe bet sizing (full Kelly leads to ruin; 1/5th Kelly worked well in NBA betting sims).
- `hybrid-lstm-transformer.js` - LSTM for temporal dependencies + transformer attention for sentiment/news + multi-head cross-feature attention.
- `sentiment-alpha.js` - LLM-based sentiment analysis for alpha generation (3% annual excess returns cited).

## Run
```
node production/drl-portfolio-manager.js
node production/fractional-kelly.js
node production/hybrid-lstm-transformer.js
node production/sentiment-alpha.js
```

## Related
- Parent: `../CLAUDE.md`.
- Wired into pipeline: `../system/trading-pipeline.js`.
- Benchmarks: `../tests/production-benchmark.js`, results in `../docs/production-benchmark-results.md`.
