# docs/adr/quantum-engine/

ADR-QE subseries for the **Quantum Engine** subsystem - ruvector's quantum-algorithm component (VQE, Grover, QAOA, surface-code error correction, tensor networks). Audience: integrators building quantum-classical hybrid workflows on ruvector.

## ADRs

- `ADR-QE-001-quantum-engine-core-architecture.md` - core architecture.
- `ADR-QE-002-crate-structure-integration.md` - Rust crate layout and integration.
- `ADR-QE-003-wasm-compilation-strategy.md` - WASM compilation strategy.
- `ADR-QE-004-performance-optimization-benchmarks.md` - perf targets and benchmarks.
- `ADR-QE-005-vqe-algorithm-support.md` - Variational Quantum Eigensolver.
- `ADR-QE-006-grover-search-implementation.md` - Grover search.
- `ADR-QE-007-qaoa-maxcut-implementation.md` - QAOA for MaxCut.
- `ADR-QE-008-surface-code-error-correction.md` - surface-code QEC.
- `ADR-QE-009-tensor-network-evaluation.md` - tensor-network evaluation.
- `ADR-QE-010-observability-monitoring.md` - monitoring/observability.
- `ADR-QE-011-memory-gating-power-management.md` - memory gating and power.
- `ADR-QE-012-mincut-coherence-integration.md` - mincut/coherence integration.
- `ADR-QE-013-deutsch-theorem-proof-verification.md` - Deutsch theorem proof verification.
- `ADR-QE-014-exotic-discoveries.md` - exotic structure discoveries.
- `ADR-QE-015-blockchain-forensics-scientific-instrument.md` - blockchain forensics application.

## Related

- `../../architecture/quantum-engine/` - DDD strategic + tactical design.
- `../../research/quantum-crypto/` - related cryptography research.
- `../../research/exotic-structure-discovery/` - exotic structure research.
