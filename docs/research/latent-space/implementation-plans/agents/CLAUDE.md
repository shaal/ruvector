# docs/research/latent-space/implementation-plans/agents/

Per-agent task decompositions for the latent-space attention swarm. Each numbered file describes the scope, deliverables, and acceptance criteria for one specialist agent.

## Agents

- `01-core-attention.md`, `02-hyperbolic-attention.md`, `03-sparse-attention.md`, `04-graph-attention.md`, `05-moe-attention.md` - attention variants.
- `06-training.md` - training pipeline.
- `07-wasm-bindings.md`, `08-napi-bindings.md` - platform bindings.
- `09-cli.md`, `10-sdk.md` - surface tools.
- `11-unit-tests.md`, `12-integration-tests.md`, `13-benchmarks.md` - test agents.
- `14-simd-optimizations.md` - SIMD agent.
- `15-cicd.md` - CICD agent.

## Related

- `../04-swarm-implementation.md` - swarm-level orchestration.
- `../../../../adr/` - shipping ADRs.
