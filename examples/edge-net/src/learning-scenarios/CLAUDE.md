# edge-net/src/learning-scenarios

Concrete learning scenarios exercised by the learning loop and tests. Each scenario contains assets / patterns / handlers.

## Important files
- `mod.rs` — scenario registry.
- `attention_patterns.rs` — attention-pattern learning scenario.
- `mcp_tools.rs` — MCP-tool-call learning.
- `sdk_integration.rs` — SDK integration scenario.

## Subdirectories
- `diverse-patterns/` — config + pattern data + setup script + TS types.
- `error_recovery/` — error-recovery scenario (Rust + sub-pattern data).
- `file_sequences/` — file-sequence scenario (Rust + sequence tracker).

## Related
- Tests: `../../tests/learning_scenarios_test.rs`.
