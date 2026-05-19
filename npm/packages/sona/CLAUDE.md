# @ruvector/sona

**SONA** — Self-Optimizing Neural Architecture. Runtime-adaptive learning engine combining LoRA, EWC++ (Elastic Weight Consolidation), and ReasoningBank, designed for LLM routers and continual-learning AI systems. Sub-millisecond learning overhead. Built as a Rust crate (`crates/sona`) exposed to Node.js via N-API (`napi-rs`); also publishes per-platform binary packages.

## Important files

- `package.json` — `@ruvector/sona` v0.1.6. Main `index.js`, types `index.d.ts`. `napi.binaryName: "sona"` with targets for linux gnu/musl, linux-arm64-gnu, darwin x64/arm64, win32 x64/arm64. Scripts: `build` (`napi build --platform --release -p ruvector-sona --manifest-path ../../../crates/sona/Cargo.toml -F napi`), `build:debug`, `test` (`node --test`), `artifacts`, `universal`, `version`. Optional deps: all `@ruvector/sona-*` per-platform packages.
- `index.js` / `index.d.ts` — `napi-rs`-generated loader that picks the right native `.node` binary (typically `SonaEngine` and related classes).
- `examples/basic-usage.js`, `examples/custom-config.js`, `examples/llm-integration.js` — runnable JS examples.
- `test/basic.test.js` — `node:test` smoke tests for `SonaEngine` (creation, custom config, trajectory recording, etc.).
- `BUILD_INSTRUCTIONS.md`, `NAPI_INTEGRATION_SUMMARY.md` — build / integration notes.
- `.npmignore` — controls what ships.
- `npm/` — sub-packages, one per platform (used as `optionalDependencies`).

## Exports

`SonaEngine` (constructor `new SonaEngine(hiddenDim)` and static `SonaEngine.withConfig({hiddenDim, microLoraRank, baseLoraRank, ...})`), `beginTrajectory(queryEmbedding)`, etc.

## Related

- Rust source: `crates/sona` (referenced in `repository.directory` and `manifest-path`).
- Per-platform packages: `npm/packages/sona/npm/{darwin-arm64, darwin-x64, linux-arm64-gnu, linux-x64-gnu, linux-x64-musl, win32-arm64-msvc, win32-x64-msvc}`.
