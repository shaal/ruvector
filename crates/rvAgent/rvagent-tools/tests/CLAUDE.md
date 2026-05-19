# rvagent-tools/tests

Per-tool integration tests.

- `ls_tests.rs`, `read_file_tests.rs`, `write_file_tests.rs`, `edit_file_tests.rs`,
  `glob_tests.rs`, `grep_tests.rs`, `execute_tests.rs`, `write_todos_tests.rs`
  — one suite per built-in tool.
- `tool_dispatch_tests.rs` — `BuiltinTool` / `AnyTool` enum dispatch.
