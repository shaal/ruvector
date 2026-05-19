# rvagent-tools

Enum-dispatched tool implementations for the rvAgent framework (ADR-103 A6).
Provides the `Tool` trait, `BuiltinTool` / `AnyTool` enum dispatch, `ToolRuntime`
context, and parallel execution helpers (ADR-103 A2).

Built-in tools:
- `LsTool` — directory listing.
- `ReadFileTool` — file read.
- `WriteFileTool` — file write.
- `EditFileTool` — in-place file edit.
- `GlobTool` — filename globbing.
- `GrepTool` — content search.
- `ExecuteTool` — shell command execution (via `LocalShellBackend`).
- `WriteTodosTool` — todo-list management.
- `TaskTool` — spawn a subagent.

## Layout

- `Cargo.toml` — lib + bench `tool_bench`. Deps: `rvagent-core`, `rvagent-backends`,
  glob, walkdir, async-trait. Dev: criterion, tempfile, mockall.
- `src/lib.rs` — re-exports each tool type.
- `src/<tool>.rs` — one file per tool listed above.
- `benches/tool_bench.rs` — Criterion bench.
- `tests/` — per-tool integration tests + `tool_dispatch_tests.rs`.

## Related

`rvagent-core` (`Tool` trait surface), `rvagent-backends` (filesystem/shell).
