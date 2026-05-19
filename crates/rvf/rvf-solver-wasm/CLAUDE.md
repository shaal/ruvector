# rvf-solver-wasm

WASM build of the RVF self-learning temporal solver. Exposes the AGI temporal-reasoning engine: PolicyKernel with two-signal Thompson Sampling, context-bucketed bandit (18 buckets = 3 range × 3 distractor × 2 noise), KnowledgeCompiler with signature-based pattern cache, speculative dual-path execution, three-loop adaptive solver (fast/medium/slow), acceptance test with training/holdout cycles, and SHAKE-256 witness chain (via `rvf-crypto`).

Target: `wasm32-unknown-unknown`, `no_std` + alloc. Uses `dlmalloc` as the global allocator and `libm` for math.

## Layout

- `Cargo.toml` — `cdylib` only. Deps: `rvf-types` (no default), `rvf-crypto` (no default), `dlmalloc` (global), `libm`, `serde`/`serde_json` (alloc-only). Release: `opt-level = "z"`.
- `src/lib.rs` — WASM exports (`rvf_solver_alloc`, `rvf_solver_free`, `rvf_solver_create`, `rvf_solver_destroy`, `rvf_solver_train`, `rvf_solver_acceptance`, `rvf_solver_result_len`, `rvf_solver_result_read`).
- `src/alloc_setup.rs` — global allocator init (dlmalloc).
- `src/engine.rs` — three-loop solver implementation.
- `src/policy.rs` — PolicyKernel + Thompson Sampling.
- `src/types.rs` — shared types (puzzles, buckets, results).

## Related

- `../rvf-wasm` — companion Cognitum-tile microkernel
- `../rvf-types`, `../rvf-crypto` — base no_std deps
- `../../ruvector-domain-expansion-wasm` — related but separate WASM surface (Meta TS + population search)
