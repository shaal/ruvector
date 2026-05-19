# delta-behavior / adr

Architecture Decision Records for the delta-behavior crate. Each ADR captures a foundational design decision and its rationale.

## Important files
- `ADR-000-DELTA-BEHAVIOR-DEFINITION.md` - canonical definition of delta-behavior and its four properties.
- `ADR-001-COHERENCE-BOUNDS.md` - design of coherence bounds and how they constrain transitions.
- `ADR-002-TRANSITION-CONSTRAINTS.md` - allowed-path transition semantics.
- `ADR-003-ATTRACTOR-BASINS.md` - attractor basin model used by the world-model and homeostasis applications.

## Related
- Domain model: `../ddd/DOMAIN-MODEL.md`. Theory: `../research/`. API: `../docs/API.md`.
- Implementations consuming these decisions: `../src/lib.rs`, `../applications/`.
