# meta-cognition-spiking-neural-network/demos

Runnable AgentDB / RuVector demonstrations, grouped by capability.

## Files
- `run-all.js` - Sequential runner. Spawns each demo in turn (vector search, attention mechanisms, self-discovery, optimization, exploration, SNN) using `child_process.spawn`.

## Subdirectories
- `attention/` - All five attention mechanisms (Multi-Head, Flash, Linear, Hyperbolic, MoE) plus a hyperbolic deep-dive.
- `exploration/` - Cognitive explorer and emergent-capability discovery scripts; persists state to `memory.bin`.
- `optimization/` - Adaptive cognitive system, performance benchmarks, and SIMD-friendly vector ops.
- `self-discovery/` - Cognitive self-discovery system with multi-attention routing; persists `memory.bin` / `enhanced-memory.bin`.
- `snn/` - SIMD-optimized spiking neural network: C++ N-API native addon, JS wrapper, examples.
- `vector-search/` - Semantic search example over a persisted `semantic-db.bin`.

## Related
- Parent: `../CLAUDE.md`.
- Architectural notes in `../docs/`.
