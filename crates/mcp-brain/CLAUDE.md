# mcp-brain

MCP (Model Context Protocol) server for the RuVector "Shared Brain". Lets Claude Code
sessions share, search, and transfer learning across sessions; knowledge is stored as
RVF cognitive containers with witness chains, Ed25519 signatures, and differential
privacy proofs. Backed by a Cloud Run service over HTTPS.

## Layout

- `Cargo.toml` — `publish = false`. Bin `mcp-brain` (`src/main.rs`) + lib. Depends on
  tokio, reqwest (rustls), sha3, regex-lite, and `ruvector-sona`.
- `src/lib.rs` — module roots and `pub use server::McpBrainServer`.
- `src/main.rs` — binary entry point. Spawns `McpBrainServer` over stdio.
- `src/server.rs` — `McpBrainServer`: JSON-RPC over stdio, dispatches the 10 brain_* tools.
- `src/tools.rs` — definitions of the 10 MCP tools (share/search/get/vote/transfer/drift/
  partition/list/delete/status).
- `src/client.rs` — HTTPS client for the Cloud Run backend.
- `src/embed.rs` — embedding helpers (sona-backed).
- `src/pipeline.rs` — request pipeline (validation, signing, witness).
- `src/types.rs` — shared DTOs.

## MCP Tools (10)

`brain_share`, `brain_search`, `brain_get`, `brain_vote`, `brain_transfer`, `brain_drift`,
`brain_partition`, `brain_list`, `brain_delete`, `brain_status`.

## Related

- `crates/sona` (alias `ruvector-sona`) — embedding/cognitive primitives.
- `crates/ruvector-mincut` — partition topology used by `brain_partition`.
