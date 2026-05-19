# RuVector

High-performance Rust-native vector database and AI agent toolkit ecosystem with AgenticDB compatibility. This is a polyglot monorepo: a large Cargo workspace of ~136 Rust crates, ~59 publishable npm packages (NAPI bindings + WASM bundles), 73 example projects, a SvelteKit chat UI, extensive ADR-driven docs, and benchmarking infrastructure.

## Top-level layout

| Dir | What's there |
|---|---|
| `crates/` | 136 Rust crates: vector DB core (HNSW, RaBitQ, ACORN), graph, attention, sublinear solvers, agent runtimes (rvAgent/rvm), LLM runtime (ruvllm), Hailo NPU, robotics, research crates. See `crates/CLAUDE.md`. |
| `examples/` | 73 example projects in Rust, TS, WASM, iOS, embedded — apps, dashboards, demos, research sketches. See `examples/CLAUDE.md`. |
| `npm/` | npm workspace: `core/` (NAPI bindings), `wasm/` (WASM build), `tests/`, and `packages/` (the 59 published packages). |
| `ui/` | `ruvocal/` — SvelteKit chat-UI fork of HuggingChat with MCP integration, intelligent LLM router, voice, WASM tools. |
| `docs/` | 123-folder doc tree: ~207 ADRs in `docs/adr/`, plus architecture, research, design notes, examples, parallel ADR namespaces. |
| `benchmarks/` | Workspace benchmarks (graph, vector-search) with analysis writeups. |
| `benches/` | Top-level cargo benches (separate from `crates/*/benches/`). |
| `tests/` | Cross-crate integration tests (agentic-jujutsu, docker-integration, distributed, rvf-integration, wasm-integration). |
| `scripts/` | Build/CI/deploy/training/publish shell scripts + a mirror of the `patches/hnsw_rs/` patch. |
| `patches/` | Vendored upstream patches (currently `hnsw_rs/` — wired as a workspace `path` dep). |
| `data/` | Training corpora (`merged_corpus.jsonl` and friends). |
| `test_models/` | Cached/test model artifacts. |
| `bench_results/` | Benchmark output (gitignored mostly). |

## How to build and run

From `package.json`:
- `npm run build` — `cargo build --release` (whole workspace).
- `npm run build:node` / `build:wasm` / `build:graph` — per-target bindings.
- `npm run build:all` — everything.
- `npm test` — `cargo test --workspace`.
- `npm run bench` — `cargo bench -p ruvector-bench`.
- `npm run lint` / `format` / `check` — cargo clippy / fmt / check.
- `npm run cli` — `cargo run -p ruvector-cli`.
- `npm run mcp` — `cargo run -p ruvector-cli --bin ruvector-mcp`.

Direct Rust build also works via `cargo build --workspace`. **Note:** the workspace `excludes` several crates (see `Cargo.toml`) — `ruvector-postgres` (needs pgrx init), `rvf/*` tree, several edge/hailo crates, large `examples/*` Cargo subtrees. Build those explicitly with `-p <name>`.

## Architecture & conventions

- **Workspace shape**: outer Cargo workspace has 100+ members; three nested sub-workspaces exist with their own `Cargo.lock`: `crates/ruvix/`, `crates/rvm/`, `crates/rvf/`. The latter two are excluded from the outer build.
- **Per-target wrappers**: a Rust crate `foo` may have sibling `foo-node` (NAPI), `foo-wasm` (wasm-pack), and an npm package at `npm/packages/foo` plus per-platform sub-packages (`foo-darwin-arm64`, etc.).
- **Research-tier lints**: many crates carry 200-line `[lints.*]` allow-blocks. Flagged in their CLAUDE.md when load-bearing.
- **ADRs**: the canonical decision log is `docs/adr/ADR-001..ADR-193` (with gaps). Two parallel ADR namespaces also exist: `docs/architecture/decisions/` and `docs/research/sublinear-time-solver/adr/` (ADR-STS-*). Numbering can collide across namespaces.
- **Iteration log**: `CHANGELOG.md` tracks per-iteration arcs (e.g., `[hailo-backend]` covers iters 133-171 for NPU acceleration on Pi 5 + AI HAT+).

## Branch notes

This repo also has feature branches `claude/remove-readme-files-o5Fje` and `main`; current work is on `my-research`.

## Per-directory documentation

**Every** subdirectory (~2,121 of them, dot-dirs excluded) has its own `CLAUDE.md` describing local purpose, key files, conventions, and pointers. Start at a top-level dir's CLAUDE.md and descend.

## Known anomalies surfaced during doc generation

- `crates/sona/` directory ships a crate named `ruvector-sona` (dir/package name mismatch).
- `npm/packages/diskann/` has a zero-byte file named `false`; declared entrypoints missing.
- `npm/packages/spiking-neural/` and `npm/packages/ruvector-wasm/` reference entrypoints not in the checkout.
- `scripts/patches/hnsw_rs/` is a near-mirror of `patches/hnsw_rs/`; only the top-level copy is wired as a path dep.
- `benchmarks/vector-search/ANALYSIS.md` notes that `QuantizedVector` distance impls in `ruvector-core` are effectively dead during HNSW search.
- `ruvllm` parent is v2.5.5 but pins per-platform optional deps at v2.0.1.
- `examples/vectorvroom/` and a couple of `models/` placeholders are empty.

## License

MIT. See `LICENSE`.
