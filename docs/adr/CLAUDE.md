# docs/adr/

Main Architecture Decision Record series for the ruvector platform. ADRs document context, decision, consequences, and status for major architectural choices across the core engine, ruvllm, RVF format, ruvagent, hailo/edge, postgres, security, and many subsystems.

Files are named `ADR-NNN-<slug>.md`. Numbering is roughly chronological; there are gaps. The series currently runs ADR-001 through ADR-193 with some intentional skips (ADR-018..023 live under `temporal-tensor-store/`).

## Reading order tips

- For newcomers, start with `ADR-001-ruvector-core-architecture.md` and `ADR-029-rvf-canonical-format.md` (the canonical binary format that supersedes earlier storage decisions).
- Coherence engine and gated-transformer: `ADR-014`, `ADR-015`, plus the `coherence-engine/` subseries.
- Delta behavior architecture: `ADR-016` plus the `delta-behavior/` subseries.
- Temporal tensor store: `ADR-017` plus the `temporal-tensor-store/` subseries (ADR-018..023).
- RVF format ecosystem: `ADR-029` .. `ADR-039`, `ADR-042` (TEE/AIDefence).
- AGI / cognitive container: `ADR-024`, `ADR-026`, `ADR-036`.
- Graph transformer architecture: `ADR-046` .. `ADR-055` (physics, biological, temporal-causal, economic, manifold layers).
- DiskANN / Vamana / RaBitQ / ACORN: `ADR-146`, `ADR-154`, `ADR-160`, `ADR-161`, `ADR-162`.
- Hailo NPU / edge: `ADR-167` .. `ADR-178`, plus `ADR-171` (ruos pi5).
- ruvllm pi cluster / continuous batching: `ADR-179` .. `ADR-182`.
- Sparse attention hardening: `ADR-183` .. `ADR-192`.
- Security / supply-chain: `ADR-007`, `ADR-012`, `ADR-042`, `ADR-172`.

## Subseries (separate ADR namespaces)

- `coherence-engine/` - ADR-CE-001 .. ADR-CE-020.
- `delta-behavior/` - ADR-DB-001 .. ADR-DB-010.
- `quantum-engine/` - ADR-QE-001 .. ADR-QE-015.
- `temporal-tensor-store/` - ADR-018 .. ADR-023.

## Related

- `../architecture/decisions/` - a parallel, older ADR namespace (numbering collides, e.g. multiple ADR-001s). Treat as historical context.
- `../architecture/` - higher-level design docs that ADRs reference.
- `../research/sublinear-time-solver/adr/` - ADR-STS series for the sublinear solver subsystem.
