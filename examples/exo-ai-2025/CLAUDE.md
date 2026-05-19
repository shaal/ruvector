# exo-ai-2025

Multi-crate research workspace for EXO-AI 2025 — a "cognitive substrate"
platform built on IIT consciousness measurement, Landauer thermodynamics,
hypergraph topology, manifold embeddings, temporal/causal memory, and
federated cognition. Includes 9 production-shaped crates, 11 standalone
research prototype crates, full TDD-style integration tests, and
extensive design docs. Research-tier code — interfaces are still in
flux; not all crates compile against the published `ruvector-*` crates.

## Top-level files

- `Cargo.toml` — workspace listing nine `crates/exo-*` members and
  patching crates.io deps to local paths (incl. `ruvector-domain-expansion`,
  `thermorust`, `ruvector-dither` from `../../crates/`).
- `Cargo.lock` — checked in (workspace lock).
- `INTEGRATION_TESTS_COMPLETE.md` — TDD plan: 28 integration tests
  across substrate/hypergraph/temporal/federation surfaces.

## Subdirectories

- `architecture/` — high-level architecture + pseudocode docs.
- `benches/` — Criterion benches for hypergraph, manifold, federation,
  temporal subsystems.
- `crates/` — the nine workspace member crates (exo-core,
  exo-hypergraph, exo-manifold, exo-temporal, exo-federation,
  exo-backend-classical, exo-exotic, exo-node, exo-wasm).
- `docs/` — API, build, security, performance, testing, OpenAPI specs.
- `report/` — comparative analysis / benchmark write-ups.
- `research/` — 11 standalone Cargo workspaces with research prototypes
  (each `Cargo.toml` declares `[workspace]` for isolation).
- `scripts/` — `run-integration-tests.sh` runner.
- `specs/` — `SPECIFICATION.md` master spec.
- `test-templates/` — per-crate test scaffolding mirrored from `tests/`.
- `tests/` — workspace-level integration tests with `common/` helpers.

## Run

```bash
# Build all workspace crates
cargo build --workspace

# Run benches
bash benches/run_benchmarks.sh

# Run integration tests
bash scripts/run-integration-tests.sh
```

## Tech stack

- Workspace deps pinned in root `[workspace.dependencies]`: serde,
  thiserror, dashmap, uuid, petgraph, tokio, criterion.
- Internal patches: `ruvector-domain-expansion`, `thermorust`,
  `ruvector-dither` (`../../crates/`).
- Optional: WASM (exo-wasm), NAPI-RS (exo-node).

## Related

- `../ecosystem-consciousness/` — uses the same IIT machinery in
  miniature
- `../../crates/ruvector-consciousness/`, `../../crates/thermorust/`
