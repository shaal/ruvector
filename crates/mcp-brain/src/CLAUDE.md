# mcp-brain/src

Source for the Shared Brain MCP server.

- `lib.rs` — module roots; re-exports `McpBrainServer`.
- `main.rs` — binary entry, runs the server over stdio.
- `server.rs` — `McpBrainServer`, JSON-RPC dispatch, transport.
- `tools.rs` — schemas/handlers for the 10 `brain_*` MCP tools.
- `client.rs` — reqwest-based client to the Cloud Run backend.
- `embed.rs` — embedding helpers via `ruvector-sona`.
- `pipeline.rs` — request pipeline (validation, Ed25519 signing, witness emission).
- `types.rs` — shared serde DTOs (Memory, Witness, Vote, etc.).
