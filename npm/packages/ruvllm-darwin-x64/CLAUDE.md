# @ruvector/ruvllm-darwin-x64

Platform-specific native (N-API) binary for `@ruvector/ruvllm` — Intel Mac (darwin-x64) with SIMD/AVX2 acceleration. Pulled in as an optional dependency by the main `@ruvector/ruvllm` package; the JS loader selects the right `*-{platform}-{arch}` package at runtime.

## Files

- `package.json` — `@ruvector/ruvllm-darwin-x64` v2.0.0. `os: ["darwin"]`, `cpu: ["x64"]`. Main and only published file: `ruvllm.darwin-x64.node`.

## Related

- Parent: `@ruvector/ruvllm` (cross-platform JS loader).
- Sibling: `npm/packages/ruvllm-win32-x64-msvc` and other `ruvllm-*` platform packages.
- Rust source: `crates/ruvector-llm` or similar.
