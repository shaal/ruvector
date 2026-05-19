# examples/seti-exotic-signals

Gallery of exotic SETI signals that boundary-first detection finds when amplitude thresholding misses them. Injects six signal classes into a 128-channel x 100-timestep spectrogram at sub-threshold amplitudes; uses temporal coherence graphs and min-cut to surface them.

## Files
- `Cargo.toml` - Binary crate `seti-exotic-signals`. Depends on `ruvector-mincut` (with `exact`), `ruvector-coherence` (with `spectral`), `rand`.
- `src/main.rs` - Generates the spectrogram, injects exotic signals, slides 20-step windows (step 5), builds inter-channel coherence graphs, runs Fiedler + min-cut, validated by 100 null permutations.

## Run
```
cargo run --release -p seti-exotic-signals
```

## Tech stack
- Rust 2021, `ruvector-coherence`, `ruvector-mincut`, `rand`.

## Related
- Sibling boundary-discovery binaries in `examples/*-boundary-discovery/`.
