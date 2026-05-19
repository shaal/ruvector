# ruvector-attention/src/sdk

High-level SDK on top of the kernels: builder, pipeline, presets — covered by `docs/SDK_GUIDE.md`.

## Files

- `mod.rs` — re-exports.
- `builder.rs` — fluent builder for assembling an attention stack.
- `pipeline.rs` — runtime pipeline that executes a built configuration.
- `presets.rs` — named presets (Flash, MLA, sparse, hyperbolic, sheaf, etc.).
