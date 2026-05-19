# edge-net

Browser-deployed distributed compute network: contributors run a WASM module that pulls tasks from a WebSocket relay, runs them on WebGPU/WebGL/Web Workers, and earns credits via QDAG-based ledger. Bundles a P2P swarm-intelligence stack with Time Crystal coordination, Neural DAG attention, MicroLoRA learning, RAC economics, and an MCP server.

## Important files
- `Cargo.toml` — crate `ruvector-edge-net` (cdylib + rlib); features `embeddings`, `neural`, `exotic`, `learning-enhanced`, `economy-enhanced`.
- `src/lib.rs` + `src/bench.rs` — WASM entry & benchmark API.
- `src/{ai,brain,compute,credits,economics,events,evolution,identity,learning,mcp,network,pikey,rac,scheduler,security,swarm,tasks,tribute,adversarial,capabilities}/` — feature modules.
- `dashboard/` — React 19 + Vite + Tailwind admin/monitoring UI.
- `relay/` — Node.js WebSocket relay (Google Cloud Functions deployable).
- `sim/` — TypeScript lifecycle simulator.
- `pkg/` — pre-built wasm-pack output, CLI (`edge-net`, `edge-net-join`), npm publish surface.
- `deploy/browser/` — embeddable HTML snippet.
- `benches/` + `tests/` — Rust + Docker-orchestrated multi-language tests.
- `docs/` — architecture, security, RAC, benchmarks, research reports.
- `run-benchmarks.sh` / `scripts/run-benchmarks.sh` — convenience scripts.

## Run / build
- WASM build: `wasm-pack build --target web --release` (or use prebuilt `pkg/`).
- Dashboard dev: `cd dashboard && npm install && npm run dev`.
- Relay: `cd relay && npm install && npm start`.
- Simulator: `cd sim && npm install && npm run simulate`.
- Rust tests: `cargo test --features full`.

## Tech stack
- WASM: `wasm-bindgen`, `web-sys` (WebGPU/WebGL/Workers/Crypto), `serde-wasm-bindgen`.
- Crypto: `ed25519-dalek`, `x25519-dalek`, `aes-gcm`, `argon2`, `zeroize`.
- Optional ruvector-* WASM crates: `ruvector-exotic-wasm`, `ruvector-learning-wasm`, `ruvector-nervous-system-wasm`, `ruvector-economy-wasm`.

## Related
- Sibling browser demos: `../wasm-vanilla`, `../wasm-react`, `../pwa-loader`, `../wasm`.
