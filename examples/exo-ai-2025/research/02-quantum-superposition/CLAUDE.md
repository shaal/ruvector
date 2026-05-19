# 02-quantum-superposition

Cognitive Amplitude Field Theory (CAFT) prototype: classical simulation
of quantum cognition with superposition, interference, and measurement
collapse modeled over complex amplitudes. Standalone workspace.

## Files

- `Cargo.toml` — standalone `[workspace]`; package
  `quantum-cognitive-superposition`. Deps: `num-complex`, `rand`,
  `rand_distr`. Three examples declared (`linda_problem`,
  `prisoners_dilemma`, `attention_collapse`).
- `RESEARCH.md`, `BREAKTHROUGH_HYPOTHESIS.md`, `EXECUTIVE_SUMMARY.md`,
  `EXPERIMENTAL_PROTOCOLS.md`, `mathematical_framework.md`,
  `VISUAL_FRAMEWORK.md`, `RESEARCH_INDEX.md`, `BIBLIOGRAPHY.bib` —
  theoretical and experimental write-ups.
- `src/lib.rs` — public surface (`quantum_cognition`).
- `src/quantum_cognitive_state.rs` — amplitude state vector.
- `src/interference_decision.rs` — interference-driven decisions.
- `src/collapse_attention.rs` — measurement-style collapse.
- `src/simd_ops.rs` — SIMD inner loops.

## Build / Run

```bash
cd examples/exo-ai-2025/research/02-quantum-superposition
cargo build --release
cargo bench
cargo run --release --example linda_problem
cargo run --release --example prisoners_dilemma
cargo run --release --example attention_collapse
```

## Related

- `../../crates/exo-core/src/backends/quantum_stub.rs`
- `../../crates/exo-exotic/src/experiments/quantum_superposition.rs`
