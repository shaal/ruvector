# rvagent-backends/tests

Integration tests for the backends.

- `filesystem_tests.rs` — `FilesystemBackend` path traversal / atomic open.
- `shell_tests.rs` — `LocalShellBackend` env sanitization + allowlist.
- `composite_tests.rs` — `CompositeBackend` prefix routing + re-validation.
- `security_tests.rs` — cross-backend security hardening (SEC-001/003/005).
- `unicode_tests.rs` — SEC-016 Unicode detection / stripping.
- `live_anthropic_test.rs` — live Anthropic Messages API smoke test (requires
  `ANTHROPIC_API_KEY`).
