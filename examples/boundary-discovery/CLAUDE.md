# examples/boundary-discovery

Synthetic-time-series demo of "boundary-first" scientific discovery: graph-structural analysis pinpoints a phase boundary that amplitude-based detectors miss. An AR(1) signal with constant marginal variance hides a hidden autocorrelation switch at sample 2000; spectral bisection of a temporal coherence graph (validated by min-cut) localizes the transition.

## Files
- `Cargo.toml` - Binary crate `boundary-discovery`, edition 2021. Depends on `ruvector-mincut` (with `exact`) and `ruvector-coherence` (with `spectral`); `rand`.
- `src/main.rs` - Generates the AR(1) regime-switching series, builds the coherence graph, runs `estimate_fiedler` and `MinCutBuilder`, validates against null permutations.

## Run
```
cargo run --release -p boundary-discovery
```

## Tech stack
- Rust 2021, RuVector crates (`ruvector-mincut`, `ruvector-coherence`), `rand 0.8`.

## Related boundary-discovery siblings
- `../cmb-boundary-discovery/` - CMB Cold Spot.
- `../frb-boundary-discovery/` - Fast Radio Burst populations.
- `../market-boundary-discovery/` - Market regime change.
- `../void-boundary-discovery/` - Cosmic voids.
- `../real-eeg-multi-seizure/` - CHB-MIT EEG seizures.
- `../seti-exotic-signals/` - Exotic SETI signals.
