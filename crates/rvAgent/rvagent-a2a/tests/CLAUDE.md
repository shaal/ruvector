# rvagent-a2a/tests

Integration tests covering the A2A spec and the r2/r3 extensions.

Spec / lifecycle:
- `task_lifecycle.rs`, `task_cancel.rs`, `dispatch_order.rs`, `card_roundtrip.rs`,
  `config_load.rs`, `artifact_kinds.rs`, `artifact_version_handshake.rs`.

Streaming / push:
- `sse_stream.rs`, `sse_backpressure.rs`, `sse_reconnect.rs`,
  `push_signing.rs`, `push_ed25519.rs`, `push_retry.rs`, `push_rejected.rs`.

r2 (identity / routing / typed artifacts):
- `card_signature.rs`, `routing_selectors.rs`, `policy_guard.rs`,
  `witness_handoff.rs`.

r3 (budget / trace / recursion):
- `budget_guard.rs`, `trace_lineage.rs`, `recursion_guard.rs`,
  `circuit_breaker.rs`, `executor_remote.rs`.

- `common/` — shared test helpers.
