# agentic-robotics-benchmarks

Criterion-based benchmark harness for the agentic-robotics runtime. Contains no library code; only `[[bench]]` targets that exercise the
core message bus, pub/sub, and async executor performance.

## Layout

- `Cargo.toml` — declares three criterion benches (`harness = false`):
  - `message_serialization`
  - `pubsub_latency`
  - `executor_performance`
- `benches/` — the actual benchmark sources (see `benches/CLAUDE.md`).

## Dependencies

- `agentic-robotics-core` (`../agentic-robotics-core`) — message types / pub/sub bus.
- `agentic-robotics-rt` (`../agentic-robotics-rt`) — async executor under test.
- `criterion` (with `html_reports`), `tokio` (full), `serde`, `serde_json`.

## Notes

- `publish = false` — benchmark crate, never released.
- Run with `cargo bench -p agentic-robotics-benchmarks`.
- See sibling `../agentic-robotics-mcp` for the MCP server integration of the same core.
