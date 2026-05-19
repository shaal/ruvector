# rvagent-wasm/src

Source for the WASM agent frontend.

- `lib.rs` — `WasmAgent`, `WasmAgentConfig`, top-level `wasm-bindgen` surface,
  `VERSION`.
- `backends.rs` — `WasmStateBackend`: in-memory virtual filesystem.
- `bridge.rs` — `BridgeMessage`, `JsModelProvider`, `to_js_value` — message
  marshalling between Rust and JS.
- `tools.rs` — `WasmToolExecutor`, `ToolRequest`, `TodoItem`/`TodoStatus` for
  the in-browser tool runtime.
- `mcp.rs` — `WasmMcpServer`: full MCP server running in the browser.
- `rvf.rs` — RVF manifest / witness bridge for WASM hosts.
- `gallery.rs` — preset / gallery agent configurations exposed to JS.
