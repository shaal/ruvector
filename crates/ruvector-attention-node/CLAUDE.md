# ruvector-attention-node

Node.js bindings (NAPI-RS) for the `ruvector-attention` crate. Exposes attention
mechanisms (dot-product, multi-head, hyperbolic, flash, linear, local/global, MoE),
training utilities (losses, optimizers, schedulers), async/batch processing, and
graph attention to Node.js.

## Layout

- `Cargo.toml` — `crate-type = ["cdylib"]`. Depends on `ruvector-attention`,
  `napi` (napi9 + async + serde-json), `napi-derive`, tokio rt-multi-thread.
  `[profile.release] lto = true, opt-level = 3, codegen-units = 1, strip = true`.
- `build.rs` — calls `napi-build`.
- `package.json` — `@ruvector/attention`, NAPI binary name `attention`. Targets:
  Windows/macOS/Linux x86_64/aarch64 (gnu + musl). Optional platform packages.
- `LICENSE` — MIT.
- `.npmignore`
- `src/` — Rust binding source (see `src/CLAUDE.md`).
- `npm/<platform>/` — per-target prebuilt `.node` artefacts + `package.json`. One
  subdir per published optional platform package.

## Exposed JS API

`AttentionConfig`, `DotProductAttention`, `MultiHeadAttention`, `FlashAttention`,
`HyperbolicAttention`, `LinearAttention`, `LocalGlobalAttention`, `MoEAttention`,
`MoEConfig`; training: `AdamOptimizer`, `AdamWOptimizer`, `SGDOptimizer`, `InfoNCELoss`,
`LocalContrastiveLoss`, `HardNegativeMiner`, `InBatchMiner`, `CurriculumScheduler`,
`LearningRateScheduler`, `SpectralRegularization`, `TemperatureAnnealing`; async:
`AttentionType`, `BatchConfig`, `BatchResult`, `BenchmarkResult`, `ParallelConfig`,
`StreamProcessor`; graph: `DualSpaceAttention`, `EdgeFeaturedAttention`,
`GraphRoPEAttention`, `RoPEConfig`. Plus `version()`.

## Related

- `crates/ruvector-attention` — underlying native crate.
