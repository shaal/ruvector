# docs/architecture/

System-level architecture and DDD design documents. Audience: contributors building or modifying core subsystems.

## Overview docs

- `SYSTEM_OVERVIEW.md` - end-to-end ruvector system overview.
- `TECHNICAL_PLAN.md` - high-level technical plan.
- `DDD.md` - domain-driven design overview across subsystems.
- `NPM_PACKAGE_ARCHITECTURE.md` - npm packaging strategy and module layout.

## Subsystem designs

- `LLM-Integration-Architecture.md` - LLM integration design.
- `bitnet-quantizer-module-design.md` - BitNet 1-bit quantizer module.
- `coherence-engine-ddd.md` - Coherence Engine DDD (see also `../adr/coherence-engine/`).
- `temporal-tensor-store-ddd.md` - Temporal tensor store DDD (see also `../adr/temporal-tensor-store/`).
- `ruvix-kernel-architecture.md` - ruvix kernel.
- `ruvltra-medium-architecture.md` - ruvltra-medium model architecture.
- `attention-exotic-ai-autonomous-systems.md` - attention/exotic-AI/autonomous-systems composition.

## Subdirs

- `decisions/` - older ADR namespace (collides numerically with `../adr/`). See its own CLAUDE.md.
- `quantum-engine/` - DDD strategic + tactical + integration for the quantum engine.

## Related

- `../adr/` - canonical ADR series.
- `../implementation/` - implementation summaries derived from these designs.
