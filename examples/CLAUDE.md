# examples/

73 example projects demonstrating RuVector features across Rust, TypeScript/JS, WASM, mobile, embedded, and full-stack apps. Most are themselves multi-file projects (apps, dashboards, demos, research sketches) — each has its own `CLAUDE.md` with run/build instructions.

## Major sub-projects

By size / scope:
- **`exo-ai-2025/`** (75 nested dirs) — multi-crate cognitive-substrate workspace; 9 production crates + 11 research workspaces + benches + reports.
- **`edge-net/`** (~60 dirs) — Rust→WASM distributed-compute network: React 19 dashboard, Node WebSocket relay (GCF-deployable), TS lifecycle simulator, pre-built `@ruvector/edge-net` npm package.
- **`vibecast-7sense/`** — full app with assets/benches/scripts/docs/tests + 9 `crates/sevensense-*` sub-crates (application/domain/infrastructure DDD).
- **`ruvLLM/`** (~41 dirs) — RuvLLM SONA + ESP32 firmware + ESP32 web flasher npm wrapper.
- **`dragnes/`** — SvelteKit dermatology app with full routes/api tree.
- **`delta-behavior/`** — ADR-driven research crate with DDD, applications, WASM SDK.
- **`scipix/`** — scientific computing app: bin/api/cache/cli/commands/math/ocr/optimize/output/preprocess/wasm, full test pyramid.
- **`prime-radiant/`** (research mirror) — category/causal/cohomology/HoTT/quantum/spectral demos.
- **`rvf/`** — RVF compiler/runtime examples + dashboard (views/three/charts/components).
- **`neural-trader/`** — full 16-subdir trading system (accounting/advanced/core/exotic/mcp/portfolio/risk/strategies/etc.).
- **`OSpipe/`** — Screenpipe AI memory system (native HTTP + WASM + dist + tests).

## Boundary-discovery cluster

A family of `*-boundary-discovery/` binaries demonstrating min-cut + Fiedler boundary detection:
`boundary-discovery`, `brain-`, `cmb-`, `earthquake-`, `frb-`, `health-`, `infrastructure-`, `market-`, `music-`, `pandemic-`, `seti-`, `void-`, `weather-`. Plus `seizure-therapeutic-sim` and `seizure-clinical-report`.

## Consciousness cluster

IIT Phi / integrated-information demos: `climate-consciousness`, `cmb-consciousness`, `ecosystem-consciousness`, `gene-consciousness`, `gw-consciousness`, `meta-cognition-spiking-neural-network`, `quantum-consciousness`.

## Smaller examples

`a2a-swarm`, `agentic-jujutsu`, `app-clip` (iOS App Clip + RVF FFI), `apify` (Apify Actors), `benchmarks`, `data` (data harvesting framework + climate/EDGAR/OpenAlex), `decompiler-dashboard`, `dna` (published as `rvdna` v0.3.0), `docs` (graph CLI/WASM docs), `edge`, `edge-full`, `esp32-mmwave-sensor` (Xtensa ESP-IDF firmware), `google-cloud`, `graph` (stub), `mincut` (workspace with 10 `[[example]]`s), `nodejs`, `onnx-embeddings*`, `pwa-loader`, `quantum-consciousness`, `real-eeg-analysis`, `real-eeg-multi-seizure`, `refrag-pipeline`, `robotics`, `rvf-desktop`, `rvf-kernel-optimized`, `seti-exotic-signals`, `spiking-network`, `spiking-neural`, `subpolynomial-time`, `temporal-attractor-discovery`, `train-discoveries`, `ultra-low-latency-sim`, `vectorvroom` (empty placeholder), `verified-applications`, `vwm-viewer`, `wasm-react`, `wasm-vanilla`, `wasm/ios`.

## Loose files at root

- `bounded_instance_demo.rs` — standalone single-file demo (not registered as a Cargo target at this level).

## Workspace status

Several large examples are **excluded from the outer `Cargo.toml` workspace** (`examples/ruvLLM`, `examples/edge-net`, `examples/data`, `examples/delta-behavior`, `examples/rvf-desktop`, `examples/ruvLLM/esp32*`) — they build standalone or with their own sub-workspace.

## Related

- Crates being demonstrated: `../crates/`.
- npm packages used by JS examples: `../npm/packages/`.
- Architecture context: `../docs/adr/`.
