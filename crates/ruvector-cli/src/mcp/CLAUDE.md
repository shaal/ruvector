# ruvector-cli/src/mcp

Model Context Protocol (MCP) implementation for the `ruvector-mcp` binary.

- `mod.rs` — module roots and re-exports.
- `protocol.rs` — JSON-RPC 2.0 message types specific to MCP.
- `handlers.rs` — tool implementations (search/insert/etc.) bridging into
  `ruvector-core` / `ruvector-graph` / `ruvector-gnn`.
- `transport.rs` — stdio + axum/hyper SSE transports.
- `gnn_cache.rs` — LRU cache (`lru` crate) for hot GNN embeddings to keep p99
  latencies bounded.
