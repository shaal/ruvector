# ruvector-decompiler/src

Source for the JS bundle decompiler.

Pipeline modules (in execution order):
- `parser.rs` — regex-driven extraction of declarations / references from minified JS.
- `graph.rs` — builds the weighted reference graph between declarations.
- `partitioner.rs` — MinCut-based module boundary detection via `ruvector-mincut`.
- `inferrer.rs` — heuristic + scored identifier renaming.
- `transformer.rs` — applies renames and module split.
- `beautifier.rs` — pretty-prints final modules.
- `witness.rs` — SHA3-256 Merkle witness emission for provenance.
- `sourcemap.rs` — source-map read/write.
- `tree.rs`, `types.rs`, `error.rs` — shared AST/types and `DecompileError`.

Optional:
- `neural.rs` (feature `neural`) — ONNX-based name inference using `ort` + `ndarray`.
- `training.rs` — offline corpus training that produces `data/*.json` patterns.
- `model_decompiler.rs`, `model_gguf.rs`, `model_safetensors.rs`, `model_types.rs`
  (feature `model`, ADR-138) — LLM weight decompiler for GGUF / safetensors.

Entry: `lib.rs` exposes `decompile(source, &DecompileConfig) -> Result<...>`.
