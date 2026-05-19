# ruvector-graph-transformer-node/npm

Per-platform sub-packages, each shipping a prebuilt NAPI `.node` binary for
`ruvector-graph-transformer-node`. The root `package.json` selects the right
one at install time via NAPI-RS's `optionalDependencies` convention.

## Sub-packages
- `darwin-arm64/`, `darwin-x64/` - macOS Apple Silicon / Intel (currently
  prebuilt).
- `linux-arm64-gnu/`, `linux-arm64-musl/` - Linux ARM64 (glibc / musl).
- `linux-x64-gnu/`, `linux-x64-musl/` - Linux x86_64.
- `win32-x64-msvc/` - Windows x86_64.

Each subdir has its own `package.json` declaring `os`/`cpu` and (when built)
a `.node` binary.
