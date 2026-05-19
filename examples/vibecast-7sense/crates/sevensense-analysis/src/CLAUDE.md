# sevensense-analysis/src

Source for the analysis bounded context.

## Files
- `lib.rs` - Crate root; documents the DDD layering and re-exports public types.
- `metrics.rs` - Shared metric helpers used across clustering / sequence analysis.

## Subdirectories
- `domain/` - Core entities, value objects, repository traits, domain events.
- `application/` - Use-case services (orchestration over domain + infra).
- `infrastructure/` - Concrete adapters (HDBSCAN, k-means, Markov, in-memory repo).

## Related
- Parent: `../CLAUDE.md`.
