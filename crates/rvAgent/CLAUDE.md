# rvAgent

Workspace umbrella for the rvAgent framework — a modular Rust agent stack
(roughly analogous to LangGraph/Deep Agents) built around a typed agent graph,
backend trait system, middleware pipeline, MCP/A2A/ACP protocol surfaces, and
parallel subagent orchestration. Each subdirectory is its own crate.

## Subcrates

- `rvagent-core/` — typed `AgentState`, config, model resolution, `AgentGraph`,
  budgets, CoW state, AGI containers (ADR-103).
- `rvagent-backends/` — filesystem / shell / composite / state / store /
  Anthropic / Gemini backends with security hardening (ADR-094, ADR-103).
- `rvagent-tools/` — enum-dispatched `Tool` impls (`ls`, `read_file`, `write_file`,
  `edit_file`, `glob`, `grep`, `execute`, `write_todos`, `task`) (ADR-103 A6).
- `rvagent-middleware/` — middleware pipeline + concrete middlewares (SONA,
  HNSW, prompt caching, retry, summarization, RVF manifest, witness, HITL,
  Unicode security).
- `rvagent-subagents/` — `SubAgentSpec`, `CompiledSubAgent`,
  `SubAgentOrchestrator`, CRDT result merge (ADR-103 C8).
- `rvagent-a2a/` — Google A2A protocol server/client (JSON-RPC, SSE, signed
  push, witness handoff) per ADR-159.
- `rvagent-acp/` — Agent Communication Protocol axum server (ADR-099, ADR-103 C6).
- `rvagent-mcp/` — Model Context Protocol server/client + skills bridge.
- `rvagent-cli/` — `rvagent` terminal coding agent + TUI.
- `rvagent-wasm/` — `WasmAgent` and `WasmMcpServer` for browser/Node.

## Top-level files

- `A7_OPTIMIZATION_REPORT.md` — performance notes (ADR-103 A7).
- `test.sh` — convenience test runner.
- `examples/` — demo shell scripts:
  - `demo_coder_agent.sh`, `demo_security_agent.sh`, `demo_tester_agent.sh`
  - `swarm/` — `hierarchical_swarm.sh`, `mesh_swarm.sh`, `pipeline_swarm.sh`
- `.ruv/` — internal workspace state.

## Related

- `crates/ruvector-sona` (used by `rvagent-middleware`), `crates/ruvector-mincut`,
  ADRs 094, 095, 099, 103, 159 under `docs/adr/`.
