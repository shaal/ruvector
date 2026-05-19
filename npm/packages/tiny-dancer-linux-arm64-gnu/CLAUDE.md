# @ruvector/tiny-dancer-linux-arm64-gnu

Platform-specific native binary package for `@ruvector/tiny-dancer` on
Linux ARM64 (glibc). Installed automatically as an optional dependency
of the main `@ruvector/tiny-dancer` package on matching hosts.

## Files
- `package.json` - Declares `os: ["linux"]`, `cpu: ["arm64"]`,
  `libc: ["glibc"]`. `main` points at
  `ruvector-tiny-dancer.linux-arm64-gnu.node`.
- `ruvector-tiny-dancer.linux-arm64-gnu.node` - Compiled napi-rs
  binary produced from `../../../crates/ruvector-tiny-dancer-node`
  (not present in source tree, copied in at publish time).

## Related
- Parent JS package: `../tiny-dancer`.
- Sibling platform packages: `../tiny-dancer-{darwin-arm64,darwin-x64,
  linux-x64-gnu,win32-x64-msvc}`.
