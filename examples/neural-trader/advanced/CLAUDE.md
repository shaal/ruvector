# neural-trader/advanced

Intermediate-to-advanced trading patterns: uncertainty quantification, live broker integration, microstructure.

## Files
- `conformal-prediction.js` - Distribution-free prediction intervals with coverage guarantees, via `@neural-trader/predictor`.
- `live-broker-alpaca.js` - Real order execution against Alpaca, position + P&L management.
- `order-book-microstructure.js` - Level 2 reconstruction, order-flow imbalance, microstructure features.

## Run
```
npm run advanced:conformal
npm run advanced:broker
npm run advanced:microstructure
```

## Related
- Parent: `../CLAUDE.md`.
- Production-grade variants: `../production/`.
