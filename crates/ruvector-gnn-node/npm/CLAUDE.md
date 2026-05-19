# ruvector-gnn-node/npm

Per-platform sub-packages, each shipping a prebuilt NAPI `.node` binary.
The root `package.json` uses `optionalDependencies` (NAPI-RS convention) to
resolve the right one at install time.

## Sub-packages
- `darwin-arm64/` - macOS Apple Silicon (currently has the only prebuilt
  `ruvector-gnn.darwin-arm64.node`).
- `darwin-x64/` - macOS Intel.
- `linux-arm64-gnu/`, `linux-arm64-musl/` - Linux ARM64 (glibc / musl).
- `linux-x64-gnu/`, `linux-x64-musl/` - Linux x86_64.
- `win32-x64-msvc/` - Windows x86_64.

Each subdir contains its own `package.json` and (when prebuilt) the `.node`
binary. Built and uploaded by CI (`.github/` workflows in the crate root).
