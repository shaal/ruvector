# crates/

Rust workspace member directory for RuVector — **136 crates** spanning vector-database core (HNSW/RaBitQ/ACORN), graph algorithms, neural/LLM kernels, attention variants, sublinear solvers, agent runtimes (rvAgent / rvm), embedded/robotics, and NAPI/WASM bindings. The root `Cargo.toml` declares which of these are members vs. `exclude`d (see top-level `Cargo.toml`).

## What lives here

By count of crates per family:
- **97 × `ruvector-*`** — the bulk of the database, indexes, attention, graph, math, kernels, plus per-target wrappers (`-node`, `-wasm`, `-cli`, `-ffi`).
- **6 × `agentic-robotics-*`** — embedded/robotics agent stack (core, embedded, rt, node, mcp, benchmarks).
- **5 × `neural-trader-*`** — trading-strategy research crates (core, coherence, replay, strategies, wasm).
- **4 × `ruqu-*`** — quantum/exotic algorithm experiments (core, algorithms, exotic, wasm).
- **3 × `ruvllm*`** — RuVector LLM runtime + WASM + sparse-attention/retrieval-diffusion research crates.
- **3 × `mcp-*`** — MCP brain server, gate, brain crates.
- **2 × `cognitum-gate-*`** — coherence-gate kernel + tilezero.
- **Singletons**: `prime-radiant` (category/cohomology/HoTT research), `ruvix` (nested workspace — its own scheduler/microkernel/aarch64-boot tree), `rvm` (nested workspace — Rust virtual machine for proof-carrying execution), `rvAgent` (multi-protocol agent umbrella), `rvf` (RVF compiler/runtime — nested workspace, excluded from outer build), `rvlite` (lightweight CRDT/Cypher/SPARQL store), `sona` (`ruvector-sona` package; learning fabric), `thermorust` and `ruos-thermal` (Pi 5 thermal supervisor), `hailort-sys` (Hailo NPU bindings).

## Top-level conventions

- Each crate has its own `CLAUDE.md` describing purpose, key modules, public API, related siblings, and any npm wrapper at `../../npm/packages/<name>`.
- Many crates carry a 200+ line `[lints.*]` allow-list — "research-tier" relaxation; flagged in individual files where load-bearing.
- WASM crates conventionally end in `-wasm`; NAPI bindings in `-node` (with sibling per-platform npm dirs under each `npm/` subfolder).
- Three crates are **nested Cargo workspaces** with their own `Cargo.lock`: `ruvix/`, `rvm/`, `rvf/`. They are partially or fully excluded from the outer workspace — see their CLAUDE.md.
- `Cargo.toml` workspace excludes documented inline: pgrx-based `ruvector-postgres`, several edge/hailo crates, `rvf` tree, large `examples/*` Cargo subtrees, and a handful of WIP crates.

## Navigation hints

- Vector DB core: `ruvector-core/`, `ruvector-acorn/`, `ruvector-rabitq/`, `ruvector-hyperbolic-hnsw/`.
- Graph: `ruvector-graph/`, `ruvector-graph-transformer/`, `ruvector-dag/`, `ruvector-mincut*`.
- Attention/transformers: `ruvector-attention/`, `ruvector-attention-unified-wasm/`, `ruvector-fpga-transformer/`, `ruvector-mincut-gated-transformer/`.
- Consensus/replication: `ruvector-raft/`, `ruvector-replication/`, `ruvector-delta-consensus/`.
- LLM runtime: `ruvllm/`, `ruvllm-cli/`, `ruvllm-wasm/`, `ruvllm_sparse_attention/`, `ruvllm_retrieval_diffusion/`.
- Edge / NPU / robotics: `ruvector-hailo*`, `agentic-robotics-*`, `ruos-thermal/`, `cognitum-gate-*`.
- Sublinear solvers: `ruvector-solver/`, `ruvector-solver-node/`, `ruvector-solver-wasm/`.
- Verification: `ruvector-verified/`, `ruvector-verified-wasm/`, `rvm/` (proof-carrying VM).
- Research: `prime-radiant/`, `ruvector-consciousness/`, `ruvector-nervous-system/`, `ruqu-*`, `neural-trader-*`.

## Related

- npm wrappers: `../npm/packages/` (per-package CLAUDE.md notes the corresponding Rust crate).
- Examples consuming these crates: `../examples/`.
- Architecture decisions: `../docs/adr/`.
