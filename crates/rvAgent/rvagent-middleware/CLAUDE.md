# rvagent-middleware

Middleware pipeline for the rvAgent DeepAgents architecture (ADR-095, ADR-103).
Defines the `Middleware` trait and `MiddlewarePipeline`, plus concrete
middlewares for filesystem, MCP bridge, memory, HNSW semantic retrieval, SONA
adaptive learning, summarization, prompt caching, retry, subagents, tool-call
patching, todo lists, tool sanitization, RVF manifest, witness, HITL
(human-in-the-loop), and Unicode security.

## Layout

- `Cargo.toml` — lib + bench `middleware_bench`.
- `src/lib.rs` — module roots + `Middleware`/`MiddlewarePipeline` core trait;
  re-exports `UnicodeSecurityChecker`, `UnicodeSecurityMiddleware`,
  `SystemPromptBuilder`, etc.
- `src/sona.rs` — SONA Adaptive Learning (three loops: instant, background,
  deep) per ADR-103 B5.
- `src/hnsw.rs` — HNSW Semantic Retrieval for skills/memory (B6, 150x-12,500x).
- `src/memory.rs` — conversation memory middleware.
- `src/filesystem.rs` — filesystem-aware middleware.
- `src/mcp_bridge.rs` — bridges MCP tools into the pipeline.
- `src/hitl.rs` — human-in-the-loop confirmation middleware.
- `src/prompt_caching.rs` — Anthropic prompt-caching.
- `src/retry.rs` — retry with backoff.
- `src/rvf_manifest.rs` — RVF manifest injection.
- `src/skills.rs` — skill loading / selection.
- `src/subagents.rs` — subagent spawning middleware.
- `src/summarization.rs` — context summarization.
- `src/todolist.rs` — todo-list tracking middleware.
- `src/tool_sanitizer.rs` — tool-input sanitization.
- `src/patch_tool_calls.rs` — patches / normalizes model tool calls.
- `src/unicode_security.rs` — `UnicodeSecurityChecker`, config, `UnicodeIssue`.
- `src/unicode_security_middleware.rs` — `UnicodeSecurityMiddleware` wrapper.
- `src/witness.rs` — witness-chain emission middleware.
- `src/utils.rs` — `append_to_system_message`, `SystemPromptBuilder`.
- `benches/middleware_bench.rs` — Criterion bench.
- `docs/UNICODE_SECURITY.md` — Unicode security middleware design notes.
- `tests/` — extensive coverage (HITL, MCP bridge, pipeline, prompt caching,
  security, summarization, Unicode security integration).

## Related

`rvagent-core`, `rvagent-mcp`, `rvagent-subagents`, `crates/ruvector-sona`,
`crates/ruvector-hnsw` (HNSW backend).
