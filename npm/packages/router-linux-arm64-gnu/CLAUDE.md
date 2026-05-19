# @ruvector/router-linux-arm64-gnu

Platform-specific native binary package for `@ruvector/router` on Linux
ARM64 (glibc). Installed automatically as an optional dependency of the
main `@ruvector/router` package on matching hosts.

## Files
- `package.json` - Declares `os: ["linux"]`, `cpu: ["arm64"]`,
  `libc: ["glibc"]`. `main` points at
  `ruvector-router.linux-arm64-gnu.node`.
- `ruvector-router.linux-arm64-gnu.node` - The compiled napi-rs binary
  (produced by `napi build` against `../../../crates/ruvector-router-ffi`;
  not present in source tree but expected at publish time).

## Related
- Parent JS package: `../router`.
- Sibling platform packages: `../router-{darwin-arm64,darwin-x64,
  linux-x64-gnu,win32-x64-msvc}`.
