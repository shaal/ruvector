# ruvbot / tests / integration

Integration tests covering cross-module behavior. Each subdir focuses
on one external collaborator.

## Subdirectories
- `core/` - Cross-cutting domain integration tests (BM25, hybrid
  search, Byzantine consensus, providers, swarm coordinator).
- `multitenancy/` - Tenant isolation invariants (ADR-002).
- `postgres/` - Persistence-layer tests against Postgres (ADR-003).
- `ruvector/` - WASM binding integration tests (ADR-006).
- `slack/` - Slack bolt integration tests.
