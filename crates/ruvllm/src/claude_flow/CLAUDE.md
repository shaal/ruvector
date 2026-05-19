# ruvllm/src/claude_flow

Claude Flow integration for RuvLTRA. Optimizes the small model for agent
routing, task classification, semantic search, and code generation, with
HNSW-powered routing (150x faster pattern search), ReasoningBank-driven
pattern learning, and a multi-phase pretraining curriculum. Also wires up
full Claude API integration with streaming and intelligent model routing
(Haiku/Sonnet/Opus by complexity).

## Files
- `mod.rs` - module wiring + public API + integration docs.
- `agent_router.rs` - task -> optimal agent type.
- `claude_integration.rs` - Claude API client with streaming.
- `model_router.rs` - Haiku/Sonnet/Opus selection by token threshold and
  complexity heuristics.
- `flow_optimizer.rs` - optimizer that consumes Claude Flow signals.
- `hnsw_router.rs` - HNSW-backed semantic router for fast pattern lookup.
- `hooks_integration.rs` - Claude Flow hook system bridge.
- `pretrain_pipeline.rs` - multi-phase curriculum pretraining pipeline.
