# rvagent-a2a/src/server

axum-based A2A server, mounted into a host binary as `server::A2aServer`.

- `mod.rs` — `A2aServer` builder + axum router wiring.
- `json_rpc.rs` — JSON-RPC 2.0 method dispatch (task.send, task.cancel, etc.).
- `sse.rs` — `text/event-stream` streaming endpoint with backpressure /
  reconnect handling (covered by `sse_*` tests).
- `push.rs` — HMAC- / Ed25519-signed push webhook delivery with retry
  (`push_*` tests).
