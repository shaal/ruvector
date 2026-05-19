# agentic-robotics-node

Node.js / TypeScript bindings (via napi-rs) for `agentic-robotics-core`, exposing the ROS2-style Publisher/Subscriber abstraction to JS. Built as a `cdylib`.

## Layout

- `Cargo.toml` — crate-type `cdylib`. Deps: `agentic-robotics-core`, `napi`/`napi-derive`, `tokio`, `serde_json`, `anyhow`. Build-deps: `napi-build`.
- `build.rs` — runs `napi_build::setup()` so napi macros generate the right TS bindings.
- `package.json` — npm package metadata (`agentic-robotics` v0.1.3) wrapping the native artifact.
- `src/lib.rs` — declares the `#[napi] AgenticNode` class. Holds maps of named `Publisher<JsonValue>` and `Subscriber<JsonValue>` under `Arc<RwLock<...>>`. JSON values bridge Rust ↔ JS payloads.

## Public API (exposed to JS)

- `AgenticNode::new(name)` — constructor
- `getName()`, `createPublisher()`, `createSubscriber()`, `publish()`, etc. (see lib.rs)

## Related

- `../agentic-robotics-core` — underlying Rust runtime
- `../agentic-robotics-embedded` — bare-metal / RTOS variant
- npm package published from this crate (see `package.json`)
