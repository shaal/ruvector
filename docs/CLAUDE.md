# docs/

Top-level documentation root for the ruvector vector-database / AI agent toolkit. Audience: contributors, integrators, and agents navigating architecture, research, and user-facing guides.

## Start here

- `INDEX.md` - master index of all documentation (canonical entry point).
- `REPO_STRUCTURE.md` - layout of the monorepo crates and packages.
- `index.html` - rendered docs landing page.

## Top-level standalone docs

- `C2-shell-execution-hardening.md`, `C8_RESULT_VALIDATION_IMPLEMENTATION.md`, `IMPLEMENTATION-C5.md` - security capability (Cn) implementation notes; see also `security/`.
- `agi-container.md`, `consciousness-api.md` - high-level surface docs for the AGI/consciousness layers.
- `moe-routing-optimization-analysis.md`, `research-openfang.md` - one-off analysis notes.

## Subdirectory map

User-facing:
- `guides/` - quick starts, tutorials, installation, WASM, AgenticDB.
- `api/` - Rust, Node.js, and Cypher API references.
- `sdk/` - SDK strategy and decision records.

Architecture and design:
- `architecture/` - system overviews, DDD docs, and a parallel `architecture/decisions/` ADR set.
- `adr/` - main numbered ADR series (ADR-001..ADR-193) plus subseries `coherence-engine/`, `delta-behavior/`, `quantum-engine/`, `temporal-tensor-store/`.
- `nervous-system/`, `sparse-inference/`, `dag/`, `gnn/`, `hnsw/`, `cnn/` - per-subsystem design docs.

Implementation and operations:
- `implementation/`, `optimization/`, `benchmarks/`, `testing/`, `code-reviews/`, `reviews/` - working notes.
- `cloud-architecture/`, `publishing/`, `development/`, `hooks/`, `integration/`, `hailo/` - ops/integration.
- `postgres/` (+ `v2/`, `zero-copy/`) - PostgreSQL extension docs.
- `project-phases/` - phase completion reports.
- `plans/` - SPARC-style implementation plans.

Domain modules:
- `ruvllm/`, `rvagent/`, `security/`, `training/`, `sql/`, `examples/`, `analysis/`.

Research (long-tail, exploratory):
- `research/` - 30+ subdirs covering RVF, sublinear solvers, latent-space, gnn-v2, quantum, etc. See `research/CLAUDE.md`.
