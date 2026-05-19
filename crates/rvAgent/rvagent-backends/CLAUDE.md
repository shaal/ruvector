# rvagent-backends

All backend implementations for rvAgent — filesystem, shell, composite, state,
store, sandbox protocols, plus Anthropic / Gemini model clients. Follows
ADR-094 (Backend Protocol & Trait System) and ADR-103 (Review Amendments).

## Backends

- `StateBackend` (`state.rs`) — ephemeral in-memory file store.
- `FilesystemBackend` (`filesystem.rs`) — local disk with security hardening
  (atomic resolve+open, path traversal protection, SEC-001).
- `LocalShellBackend` (`local_shell.rs`) — filesystem + shell execution with
  env sanitization (SEC-005) and command allowlist.
- `CompositeBackend` (`composite.rs`) — path-prefix routing to sub-backends with
  post-strip re-validation (SEC-003).
- `StoreBackend` (`store.rs`) — persistent key-value storage; RVF variant in
  `rvf_store.rs`.

## Security (ADR-103)

Implemented across `security.rs`, `unicode_security.rs` (SEC-016 detection /
stripping), and within each backend (atomic path checks, env sanitization,
literal grep mode SEC-021).

## Layout

- `Cargo.toml` — lib.
- `src/lib.rs` — module roots and `pub use` of backend types + `AnthropicClient`
  etc.
- `src/protocol.rs` — shared backend trait / message protocol.
- `src/anthropic.rs`, `src/gemini.rs` — model API clients.
- `src/sandbox.rs` — sandbox enforcement.
- `src/utils.rs` — shared helpers.
- `benches/backend_bench.rs` — Criterion bench.
- `tests/` — `filesystem_tests.rs`, `composite_tests.rs`, `shell_tests.rs`,
  `security_tests.rs`, `unicode_tests.rs`, `live_anthropic_test.rs`.

## Related

- `crates/rvAgent/rvagent-core` (`BackendConfig`, `SecurityPolicy`),
  `crates/rvAgent/rvagent-tools` (which consumes these backends).
