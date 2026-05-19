# rvagent-backends/src

Source for backend implementations (ADR-094, ADR-103).

- `lib.rs` — module roots, re-exports `AnthropicClient`, `CompositeBackend`,
  `BackendRef`, `FilesystemBackend`, `LocalShellBackend`, `CommandAllowlist`,
  `LocalShellConfig`, etc.
- `protocol.rs` — common backend trait + message protocol.
- `state.rs` — `StateBackend` (in-memory).
- `filesystem.rs` — `FilesystemBackend` with path-traversal hardening (SEC-001).
- `local_shell.rs` — `LocalShellBackend` with env sanitization (SEC-005),
  `CommandAllowlist`, `LocalShellConfig`.
- `composite.rs` — `CompositeBackend` with prefix re-validation (SEC-003).
- `store.rs` — persistent key-value `StoreBackend`.
- `rvf_store.rs` — RVF-flavored persistent store.
- `sandbox.rs` — sandbox enforcement primitives.
- `anthropic.rs` — `AnthropicClient` (Messages API).
- `gemini.rs` — Gemini model client.
- `security.rs` — shared security helpers.
- `unicode_security.rs` — SEC-016 Unicode detection / stripping.
- `utils.rs` — shared utilities.
