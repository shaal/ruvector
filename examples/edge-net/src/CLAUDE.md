# edge-net/src

Rust source for the `ruvector-edge-net` cdylib/rlib. Compiled to WASM and consumed by browser contributors via `../pkg/`.

## Important files
- `lib.rs` — WASM entry, public exports, feature gates.
- `bench.rs` — benchmark hooks (`bench` feature).

## Subdirectories (feature modules)
- `ai/` — attention, federated, LoRA, memory, router, plus Sona reasoning bank.
- `brain/` — high-level brain orchestrator.
- `capabilities/` — capability declarations.
- `compute/` — execution backends: SIMD, Web Workers, WebGL, WebGPU + WGSL/GLSL shaders.
- `credits/` — credit ledger + QDAG.
- `economics/` — AMM, brain rewards, reputation.
- `events/` — internal event bus.
- `evolution/` — evolutionary search.
- `identity/` — cryptographic identity.
- `learning/` — online learning loop.
- `learning-scenarios/` — concrete learning scenarios (attention, MCP, SDK, error recovery, file sequences, diverse patterns).
- `mcp/` — MCP server (handlers, protocol, transport).
- `network/` — P2P, semantic, protocols.
- `pikey/` — PiKey identity scheme.
- `rac/` — RAC economics + axioms.
- `scheduler/` — task scheduler.
- `security/` — security primitives.
- `swarm/` — collective intelligence (consensus, stigmergy).
- `tasks/` — task lifecycle.
- `tribute/` — tribute / payout flow.
- `adversarial/` — adversarial scenarios.

## Build
- `cargo build --release --target wasm32-unknown-unknown --features full` (or use `wasm-pack`).
