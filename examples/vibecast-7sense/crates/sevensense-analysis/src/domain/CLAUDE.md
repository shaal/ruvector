# sevensense-analysis/src/domain

Domain layer of the analysis bounded context.

## Files
- `mod.rs` - Aggregates and re-exports.
- `entities.rs` - Core domain entities (clusters, prototypes, motifs, sequences).
- `value_objects.rs` - Value objects (similarity scores, cluster ids, distance metrics).
- `repository.rs` - Repository traits implemented by the infrastructure layer.
- `events.rs` - Domain events emitted during analysis (cluster updated, anomaly detected, ...).

## Related
- Application layer: `../application/`.
- Infrastructure: `../infrastructure/`.
