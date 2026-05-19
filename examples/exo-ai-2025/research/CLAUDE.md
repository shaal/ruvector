# exo-ai-2025/research

Eleven standalone research-prototype crates, each its own Cargo
workspace (`[workspace]` in their `Cargo.toml`) so they compile in
isolation from the parent workspace. Each prototype is paired with one
or more design docs (`RESEARCH.md`, `BREAKTHROUGH_HYPOTHESIS.md`,
`EXECUTIVE_SUMMARY.md`, etc.).

## Subdirectories

- `01-neuromorphic-spiking/` — bit-parallel spiking neural networks for
  consciousness computation.
- `02-quantum-superposition/` — Cognitive Amplitude Field Theory
  (CAFT), classical simulation of quantum cognition.
- `03-time-crystal-cognition/` — discrete time crystals as working
  memory.
- `04-sparse-persistent-homology/` — sub-cubic persistent homology
  with SIMD acceleration.
- `05-memory-mapped-neural-fields/` — petabyte-scale neural fields via
  mmap + tiered storage.
- `06-federated-collective-phi/` — distributed IIT Phi with CRDTs and
  Byzantine consensus.
- `07-causal-emergence/` — O(log n) emergence-detection framework
  (HCC).
- `08-meta-simulation-consciousness/` — O(N^3) Phi for ergodic systems.
- `09-hyperbolic-attention/` — hyperbolic attention networks for
  hierarchical reasoning.
- `10-thermodynamic-learning/` — physics-based learning at Landauer
  limits.
- `11-conscious-language-interface/` — ruvLLM + spiking +
  ruvector consciousness integration.
- `docs/` — markdown overview per research crate, indexed at
  `research/PAPERS.md`.
- `PAPERS.md`, `RUST_LIBRARIES.md`, `TECHNOLOGY_HORIZONS.md` —
  bibliographic indices.

## Build

```bash
# Each subdir is its own workspace; cd into one first:
cd 01-neuromorphic-spiking && cargo build
```

## Related

- `../crates/exo-exotic/src/experiments/` — production-shape
  re-imaginings of several of these prototypes
- `../report/` — comparative analysis
