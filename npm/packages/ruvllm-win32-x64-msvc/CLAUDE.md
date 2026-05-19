# @ruvector/ruvllm-win32-x64-msvc

Platform-specific native (N-API) binary for `@ruvector/ruvllm` — Windows x64 (MSVC) with SIMD/AVX2 acceleration. Pulled in as an optional dependency by the main `@ruvector/ruvllm` package; the JS loader selects the right `*-{platform}-{arch}` package at runtime.

## Files

- `package.json` — `@ruvector/ruvllm-win32-x64-msvc` v2.0.0. `os: ["win32"]`, `cpu: ["x64"]`. Main and only published file: `ruvllm.win32-x64-msvc.node`.

## Related

- Parent: `@ruvector/ruvllm` (cross-platform JS loader).
- Sibling: `npm/packages/ruvllm-darwin-x64` and other `ruvllm-*` platform packages.
