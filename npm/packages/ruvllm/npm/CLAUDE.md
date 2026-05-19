# npm/

Per-platform NAPI sub-packages for `@ruvector/ruvllm`. Each subdirectory is published as its own npm package containing only a `package.json` (the `.node` binary is dropped here at publish time by `napi build`).

Subdirectories:

- `darwin-arm64/`, `darwin-x64/` — macOS arm64/x64.
- `linux-arm64-gnu/`, `linux-x64-gnu/` — Linux arm64/x64 (glibc).
- `win32-x64-msvc/` — Windows x64 (MSVC).

These are listed in the parent `package.json`'s `optionalDependencies` (versions pinned to `2.0.1` in the current manifest, even though the parent is at 2.5.5).
