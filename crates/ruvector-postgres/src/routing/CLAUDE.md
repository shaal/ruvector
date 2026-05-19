# ruvector-postgres/src/routing

Tiny Dancer Routing — neural-powered dynamic agent routing with FastGRNN for adaptive decision-making.

## Files

- `mod.rs` — Re-exports `Agent`, `AgentRegistry`, `AgentType`, `FastGRNN`, `Router`, `RoutingDecision`, `OptimizationTarget`, `RoutingConstraints`.
- `fastgrnn.rs` — `FastGRNN` cell + forward pass.
- `agents.rs` — `Agent`, `AgentRegistry`, `AgentType` enum.
- `router.rs` — `Router` with `OptimizationTarget`, `RoutingConstraints`, `RoutingDecision`.
- `operators.rs` — pgrx SQL operator surface.

## Pointers

- Backbone: `ruvector-tiny-dancer-core` (also bound for WASM/Node via `ruvector-tiny-dancer-wasm` and `ruvector-router-ffi`).
- See `../../docs/ROUTING_QUICK_REFERENCE.md`, `../../docs/TINY_DANCER_ROUTING.md`.
