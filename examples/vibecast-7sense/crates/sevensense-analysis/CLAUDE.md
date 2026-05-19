# sevensense-analysis

Analysis bounded context for 7sense: clustering, motif detection, Markov sequence analysis, anomaly detection over bioacoustic embeddings. DDD hexagonal layout.

## Files
- `Cargo.toml` - Crate `sevensense-analysis`. Depends on `sevensense-core`, `sevensense-vector`, `tokio`, plus the analysis deps.
- `src/lib.rs` - Crate root, re-exports.
- `src/metrics.rs` - Shared metric helpers.
- `src/domain/` - Entities, value objects, repository traits, events.
- `src/application/` - Application services orchestrating clustering / sequence work.
- `src/infrastructure/` - HDBSCAN, k-means, Markov chain implementations, in-memory repository.

## Build
```
cargo build -p sevensense-analysis
```

## Related
- Used by: `sevensense-interpretation`, `sevensense-api`.
- Benchmarks: `../../benches/clustering_benchmark.rs`.
