# docs/research/

Long-tail research and exploratory write-ups that inform (but are not yet promoted to) ADRs. Each subdir focuses on a specific topic - from algorithms (mincut, spectral sparsification, sublinear-time solvers) to applications (climate-consciousness, gene-consciousness, agentic-robotics, DrAgnes/dermatology) to internal experiments (RVF format spec, ruvllm rebuilds, latent-space attention research).

## Subdirs (by theme)

Algorithms / theory:
- `mincut/`, `spectral-sparsification/`, `sublinear-time-solver/` (with `adr/` and `ddd/`), `miller-rabin-optimizations/`, `quantization-edge/`, `latent-space/` (with `implementation-plans/` -> `agents/`).

Format / runtime:
- `rvf/` (with `spec/`, `wire/`, `crypto/`, `microkernel/`, `profiles/`, `benchmarks/`) - RVF (ruvector format) spec & runtime.
- `wasm-integration-2026/`, `pglite/`, `knowledge-export/`.

Models / language:
- `ruvllm/`, `ruvm/`, `models/`, `gnn-v2/`, `cnn/`, `cognitive-frontier/`, `dspy/`.

Applications / domains:
- `climate-consciousness/`, `gene-consciousness/`, `pi-brain/`, `seizure-prediction/`, `DrAgnes/` (dermatology), `agentic-robotics/`, `ruvagent-gemini-grounding/`, `rvagent-gemini-grounding/` (same theme).

Discovery / vision:
- `exotic-structure-discovery/`, `quantum-crypto/`, `rv2/` (long-term vision), `federated-rvf/`, `sparql/`.

Datalakes / storage:
- `ruLake/`.

Decompilation / source archeology:
- `claude-code-rvsource/` - reverse engineering of claude-code v2.1 (extracted JS sources, RVF artifacts, version trees). Large; mostly machine-extracted.

## Related

- `../adr/` - decisions promoted from this research.
- `../architecture/` - higher-level design synthesis.
- `../plans/` - SPARC plans derived from research.
