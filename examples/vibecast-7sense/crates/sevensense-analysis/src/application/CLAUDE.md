# sevensense-analysis/src/application

Application layer of the analysis bounded context.

## Files
- `mod.rs` - Re-exports the application surface.
- `services.rs` - Orchestrating services that combine domain repositories with infrastructure implementations (HDBSCAN/k-means/Markov) to expose clustering, motif detection, and sequence analysis use cases.

## Related
- Domain types: `../domain/`.
- Adapters: `../infrastructure/`.
