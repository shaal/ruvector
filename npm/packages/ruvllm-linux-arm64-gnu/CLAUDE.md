# @ruvector/ruvllm-linux-arm64-gnu

Standalone publishable per-platform NAPI package: Linux arm64 (glibc) binary for `@ruvector/ruvllm`. SIMD/NEON acceleration.

- `package.json` — v2.0.0; `os: ["linux"]`, `cpu: ["arm64"]`, libc glibc, main `ruvllm.linux-arm64-gnu.node`.

The `.node` binary is produced by `napi build` in the parent `npm/packages/ruvllm/` package and dropped here at publish time.

## Related

- Parent: `npm/packages/ruvllm/`.
- Sibling stub in same role: `npm/packages/ruvllm/npm/linux-arm64-gnu/` (alternate location for the same content).
