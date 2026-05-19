# rvagent-tools/src

Built-in tool implementations (ADR-103 A6).

- `lib.rs` — module roots and re-exports of each tool type.
- `ls.rs` — `LsTool`.
- `read_file.rs` — `ReadFileTool`.
- `write_file.rs` — `WriteFileTool`.
- `edit_file.rs` — `EditFileTool`.
- `glob.rs` — `GlobTool` (uses `glob` crate).
- `grep.rs` — `GrepTool`.
- `execute.rs` — `ExecuteTool` (shell execution via `rvagent-backends`).
- `write_todos.rs` — `WriteTodosTool`.
- `task.rs` — `TaskTool` (spawn a subagent).

All tools implement the `Tool` trait from `rvagent-core`; dispatched via the
`BuiltinTool`/`AnyTool` enums for monomorphisation.
