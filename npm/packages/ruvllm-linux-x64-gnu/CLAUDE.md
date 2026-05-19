# @ruvector/ruvllm-linux-x64-gnu

Platform-specific native binary package for `@ruvector/ruvllm` on
Linux x86_64 (glibc). Built with AVX2 SIMD acceleration. Installed
automatically as an optional dependency of the main `@ruvector/ruvllm`
package on matching hosts.

## Files
- `package.json` - Declares `os: ["linux"]`, `cpu: ["x64"]`,
  `libc: ["glibc"]`. `main` points at `ruvllm.linux-x64-gnu.node`.
- `ruvllm.linux-x64-gnu.node` - Compiled napi-rs binary produced from
  `../../../crates/ruvllm` (copied in at publish time).

## Related
- Sibling platform packages: `../ruvllm-{darwin-arm64,darwin-x64,
  linux-arm64-gnu,win32-x64-msvc}`.
- Rust crate: `../../../crates/ruvllm`.
