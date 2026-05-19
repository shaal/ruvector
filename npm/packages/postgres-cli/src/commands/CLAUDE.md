# postgres-cli/src/commands

Per-feature command modules used by the `ruvector-pg` / `rvpg` CLI and re-exported by `src/index.ts`. Each module wires `commander` subcommands to calls into `RuVectorClient`.

## Files

- `vector.ts` — `VectorCommands`: insert, search, similarity, dense/sparse vector ops.
- `attention.ts` — `AttentionCommands`: 39 attention mechanisms (scaled-dot, multi-head, flash, etc.).
- `gnn.ts` — `GnnCommands`: GNN layers (GCN, GraphSAGE, GAT).
- `graph.ts` — `GraphCommands`: graph ops + Cypher-style traversals.
- `learning.ts` — `LearningCommands`: self-learning / ReasoningBank operations.
- `benchmark.ts` — `BenchmarkCommands`: runs the SQL benchmarks in `../../benchmarks/`.
- `sparse.ts` — sparse vector / BM25 / TF-IDF / SPLADE commands.
- `hyperbolic.ts` — hyperbolic geometry (Poincare, Lorentz) embedding commands.
- `routing.ts` — Tiny Dancer agent routing commands.
- `quantization.ts` — vector quantization commands.
- `install.ts` — installer / extension setup.

Each `.ts` has compiled `.js`, `.d.ts`, and `.map` siblings.
