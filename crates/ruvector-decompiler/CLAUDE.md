# ruvector-decompiler

SOTA JavaScript bundle decompiler. Reverses minified JS into human-readable modules
using a five-stage pipeline: regex parsing -> reference graph -> MinCut module
partitioning -> name inference -> SHA3-256 Merkle witness chain. Optional neural
name inference via ONNX (`ort`) and LLM weight decompilation (GGUF / safetensors).

## Layout

- `Cargo.toml` — lib. Deps: `ruvector-mincut`, regex, sha3, serde, rayon (default
  feature `parallel`), memchr; optional `ort` + `ndarray` for `neural` feature.
  Features: `parallel`, `wasm` (passes through to ruvector-mincut), `neural`, `model`.
  Two `[[bench]]`: `bench_parser`, `bench_pipeline`.
- `src/lib.rs` — module roots; documents the 5-stage pipeline and public `decompile`.
- `src/parser.rs` — regex-based JS declaration extraction.
- `src/graph.rs` — weighted cross-declaration reference graph.
- `src/partitioner.rs` — MinCut-based module boundary detection.
- `src/inferrer.rs` — heuristic + scored name inference.
- `src/transformer.rs` — output transformation / renaming.
- `src/beautifier.rs` — formatted source emission.
- `src/sourcemap.rs` — source-map I/O.
- `src/witness.rs` — Merkle witness chain emission.
- `src/training.rs` — pattern training for the inferrer (feeds `data/`).
- `src/neural.rs` — ONNX-based neural name inference (feature `neural`).
- `src/tree.rs`, `src/types.rs`, `src/error.rs` — supporting types.
- `src/model_decompiler.rs`, `src/model_gguf.rs`, `src/model_safetensors.rs`,
  `src/model_types.rs` — LLM model weight decompiler (ADR-138, feature `model`).
- `benches/bench_parser.rs`, `benches/bench_pipeline.rs` — Criterion benches.
- `tests/integration.rs`, `tests/ground_truth.rs`, `tests/real_world.rs`,
  `tests/model_decompiler.rs` — integration + corpus tests.
- `examples/run_on_cli.rs` — example decompiling a sample bundle.
- `data/claude-code-patterns.json` — name-inference pattern corpus (MCP-related
  property/context patterns).

## Public API

`decompile`, `DecompileConfig` (+ result `modules` with `name`/`source`), pipeline
modules listed above.

## Related

- `crates/ruvector-mincut` — graph partitioning backend.
