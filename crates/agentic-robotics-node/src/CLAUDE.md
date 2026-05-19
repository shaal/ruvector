# agentic-robotics-node/src

Sole source dir for the NAPI bindings.

## Files

- `lib.rs` — `AgenticNode` struct annotated with `#[napi]`, holds `publishers` / `subscribers` keyed by topic string, all wrapped in `Arc<RwLock<HashMap<_, Arc<...>>>>`. Bridges `agentic_robotics_core::{Publisher, Subscriber}` with `serde_json::Value` payloads so JS can produce/consume arbitrary JSON.
