# docs/adr/coherence-engine/

ADR-CE subseries for the **Coherence Engine** subsystem - a sheaf-Laplacian-based mechanism for tracking and gating cognitive coherence in ruvector. Decision context: the coherence engine was carved out of the main ADR series after ADR-014 introduced it, and ADR-015 paired it with the gated transformer.

## ADRs

- `ADR-CE-001-sheaf-laplacian-coherence.md` - core math: sheaf Laplacian over a knowledge graph.
- `ADR-CE-002-incremental-computation.md` - incremental updates instead of full recomputation.
- `ADR-CE-003-hybrid-storage.md` - hybrid in-memory + persistent storage layout.
- `ADR-CE-004-signed-event-log.md` - audit log with cryptographic signatures.
- `ADR-CE-005-governance-objects.md` - governance object model.
- `ADR-CE-006-compute-ladder.md` - escalating compute tiers for coherence checks.
- `ADR-CE-007-threshold-autotuning.md` - adaptive thresholds.
- `ADR-CE-008-multi-tenant-isolation.md` - per-tenant isolation.
- `ADR-CE-009-single-coherence-object.md` - one coherence object per session.
- `ADR-CE-010-domain-agnostic-substrate.md` - keep the substrate domain-agnostic.
- `ADR-CE-011-residual-contradiction-energy.md` - "energy" measure of contradiction.
- `ADR-CE-012-gate-refusal-witness.md` - refusal witness for gated outputs.
- `ADR-CE-013-not-prediction.md` - coherence is gating, not prediction.
- `ADR-CE-014-reflex-lane-default.md` - default reflex lane behavior.
- `ADR-CE-015-adapt-without-losing-control.md` - adaptation without losing control invariants.
- `ADR-CE-016-ruvllm-coherence-validator.md` - integration with ruvllm.
- `ADR-CE-017-unified-audit-trail.md` - unified audit trail across components.
- `ADR-CE-018-pattern-restriction-bridge.md` - pattern restriction bridge.
- `ADR-CE-019-memory-as-nodes.md` - memory modeled as graph nodes.
- `ADR-CE-020-confidence-from-energy.md` - derive confidence from residual energy.

## Related

- `../ADR-014-coherence-engine.md`, `../ADR-015-coherence-gated-transformer.md` - parent ADRs.
- `../../architecture/coherence-engine-ddd.md` - DDD design.
