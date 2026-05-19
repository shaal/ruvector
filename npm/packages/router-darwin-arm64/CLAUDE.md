# @ruvector/router-darwin-arm64

Platform-specific native (N-API) binary for `@ruvector/router` — macOS Apple Silicon (`darwin-arm64`). Pulled in as an optional dependency by the main router package; the JS loader picks the right `*-{platform}-{arch}` package at runtime.

## Files

- `package.json` — `@ruvector/router-darwin-arm64` v0.1.30. `os: ["darwin"]`, `cpu: ["arm64"]`. Main and only published file: `ruvector-router.darwin-arm64.node`.

## Related

- Parent: `@ruvector/router` (the cross-platform JS loader).
- Sibling platform packages: `npm/packages/router-linux-x64-gnu` and other `router-*` directories.
