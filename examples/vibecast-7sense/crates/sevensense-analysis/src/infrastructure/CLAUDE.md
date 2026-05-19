# sevensense-analysis/src/infrastructure

Infrastructure adapters that implement the domain repository traits.

## Files
- `mod.rs` - Wires concrete adapters.
- `hdbscan.rs` - HDBSCAN clustering implementation.
- `kmeans.rs` - K-means clustering implementation.
- `markov.rs` - Markov chain transition matrices and entropy computations for sequence analysis.
- `memory_repository.rs` - In-memory repository implementation (for tests / single-node deployments).

## Related
- Domain: `../domain/`.
- Use cases: `../application/`.
