# examples/market-boundary-discovery

Boundary-first discovery applied to market regime changes: detects a hidden "bull-volatile" regime fracture in the correlation structure ~100 days before the price index drops. Uses 10 synthetic assets across 500 days with three regimes (bull-quiet, bull-volatile, crash).

## Files
- `Cargo.toml` - Binary crate `market-boundary-discovery`. Depends on `ruvector-mincut` (with `exact`), `ruvector-coherence` (with `spectral`), `rand`.
- `src/main.rs` - Generates correlated returns, slides 10-day windows (50 windows total), builds correlation graphs per window, and detects the structural boundary via Fiedler + mincut; 80 null permutations.

## Run
```
cargo run --release -p market-boundary-discovery
```

## Tech stack
- Rust 2021, `ruvector-coherence`, `ruvector-mincut`, `rand`.

## Related
- Sibling boundary-discovery binaries.
- For full trading workflows see `examples/neural-trader/`.
