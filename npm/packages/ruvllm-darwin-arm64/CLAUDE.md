# @ruvector/ruvllm-darwin-arm64

Platform-specific native binary package for `@ruvector/ruvllm` on
macOS Apple Silicon (darwin-arm64). Built with NEON SIMD acceleration
and loaded automatically as an optional dependency of the main
`@ruvector/ruvllm` package.

## Files
- `package.json` - Declares `os: ["darwin"]`, `cpu: ["arm64"]`.
  `main` points at `ruvllm.darwin-arm64.node`.
- `ruvllm.darwin-arm64.node` - Compiled napi-rs binary produced from
  `../../../crates/ruvllm` (not present in source tree, copied in at
  publish time).

## Related
- Parent JS package: not in this chunk — see `../ruvllm/` if present.
- Sibling platform packages: `../ruvllm-{darwin-x64,linux-x64-gnu,
  linux-arm64-gnu,win32-x64-msvc}`.
- Rust crate: `../../../crates/ruvllm`.
