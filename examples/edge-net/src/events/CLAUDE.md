# edge-net/src/events

Internal event bus used by other modules to publish/subscribe across the WASM runtime.

## Important files
- `mod.rs` — event types and bus implementation (uses `typed-arena` + `parking_lot`).
