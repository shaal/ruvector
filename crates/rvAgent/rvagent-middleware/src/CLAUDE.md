# rvagent-middleware/src

Source for the middleware pipeline (ADR-095, ADR-103).

- `lib.rs` — `Middleware` trait + `MiddlewarePipeline`, module roots, re-exports
  for Unicode security and `SystemPromptBuilder`.
- `sona.rs` — SONA Adaptive Learning (B5).
- `hnsw.rs` — HNSW Semantic Retrieval for skills / memory (B6).
- `memory.rs` — conversation memory.
- `filesystem.rs` — filesystem-aware middleware.
- `mcp_bridge.rs` — bridges MCP tools into the pipeline.
- `hitl.rs` — human-in-the-loop confirmation.
- `prompt_caching.rs` — Anthropic prompt-cache markers.
- `retry.rs` — retry with backoff.
- `rvf_manifest.rs` — RVF manifest injection.
- `skills.rs` — skill loading / selection.
- `subagents.rs` — subagent spawn middleware.
- `summarization.rs` — context summarization.
- `todolist.rs` — todo-list tracking.
- `tool_sanitizer.rs` — tool-input sanitization.
- `patch_tool_calls.rs` — normalizes model tool calls.
- `unicode_security.rs` + `unicode_security_middleware.rs` — SEC-016 detection
  and middleware wrapper.
- `witness.rs` — witness-chain emission.
- `utils.rs` — `append_to_system_message`, `SystemPromptBuilder`.
