# market-boundary-discovery/src

Source for the market-regime boundary-discovery binary.

## Files
- `main.rs` - Generates `N_ASSETS=10` time series over `N_DAYS=500` with three regimes (bull-quiet ends day 150, bull-volatile ends day 250, crash after), runs sliding windows of `WIN=10` days, builds per-window correlation graphs and locates the structural boundary using `estimate_fiedler` + `MinCutBuilder`; `NULL_N=80` permutations for significance, seed 42.

## Related
- Parent: `../CLAUDE.md`.
- Trading-oriented examples in `examples/neural-trader/`.
