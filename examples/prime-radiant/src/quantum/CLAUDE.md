# prime-radiant/src/quantum

Quantum/topology module: quantum states, density matrices, quantum channels, simplicial complexes, persistent homology, topological codes/invariants.

## Files

- `mod.rs` - Module surface.
- `complex_matrix.rs` (~30KB) - Complex-valued matrix primitives.
- `quantum_state.rs` (~21KB) - Pure state vectors and operations.
- `density_matrix.rs` (~22KB) - Density matrices and mixed states.
- `quantum_channel.rs` (~22KB) - Quantum channels / CPTP maps.
- `simplicial_complex.rs` (~27KB) - Simplicial complex data structure.
- `persistent_homology.rs` (~22KB) - Persistent homology computation.
- `topological_code.rs` (~32KB) - Topological quantum codes.
- `topological_invariant.rs` - Computed invariants.
- `coherence_integration.rs` - Integration with the coherence subsystem.

## Related

- ADR: `../../docs/adr/ADR-006-quantum-topology.md`.
- Benches: `../../benches/quantum_bench.rs`, `../../benches/quantum_solver_bench.rs`.
