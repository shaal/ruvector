# ruvllm/src/context

Intelligent context management with semantic memory, pruning, and bridges to
Claude Flow's memory system. Composes agentic, episodic, semantic, and
working memories under a single `IntelligentContextManager`.

## Files
- `mod.rs` - public API + architecture diagram.
- `context_manager.rs` - top-level `IntelligentContextManager` (the glue).
- `agentic_memory.rs` - agentic memory backend.
- `episodic_memory.rs` - episodic (time-ordered) memory.
- `semantic_cache.rs` - HNSW-backed semantic cache.
- `working_memory.rs` - short-term working memory.
- `claude_flow_bridge.rs` - sync/proxy to Claude Flow's memory system.
