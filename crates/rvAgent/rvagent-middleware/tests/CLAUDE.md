# rvagent-middleware/tests

Integration tests for individual middlewares and the full pipeline.

- `pipeline_tests.rs` — pipeline composition and ordering semantics.
- `hitl_tests.rs` — human-in-the-loop flow.
- `mcp_bridge_tests.rs` — MCP bridge middleware.
- `prompt_caching_tests.rs` — Anthropic prompt cache markers.
- `summarization_tests.rs` — context summarization.
- `security_tests.rs`, `security_middleware_tests.rs` — security middlewares.
- `unicode_security_integration.rs` — full Unicode security middleware stack
  (SEC-016).
