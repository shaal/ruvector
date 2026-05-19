# rvagent-wasm

Browser / Node.js WASM frontend for rvAgent. Provides `WasmAgent`, a
`wasm-bindgen` exported agent that runs entirely in the browser or Node.js with
an in-memory virtual filesystem (`WasmStateBackend`) and delegates model calls
to JavaScript via `JsModelProvider`. Also ships `WasmMcpServer` so an MCP
server can run in the browser with no separate process.

## Layout

- `Cargo.toml` — `cdylib` (wasm-bindgen).
- `src/lib.rs` — `WasmAgent`, `WasmAgentConfig`, glue between the JS host and
  the Rust agent loop; module roots.
- `src/backends.rs` — `WasmStateBackend` (in-memory virtual filesystem).
- `src/bridge.rs` — `BridgeMessage`, `JsModelProvider`, `to_js_value` —
  JS<->WASM message glue.
- `src/tools.rs` — `WasmToolExecutor`, `ToolRequest`, `TodoItem`/`TodoStatus`
  WASM-friendly tool surface.
- `src/mcp.rs` — `WasmMcpServer`, in-browser MCP server.
- `src/rvf.rs` — RVF manifest / witness bridge for WASM.
- `src/gallery.rs` — preset / gallery configurations.

## Related

`rvagent-core` (`AgentState`, prompts), `rvagent-mcp` (protocol), `rvagent-tools`
(tool surface mirrored).
