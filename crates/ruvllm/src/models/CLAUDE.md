# ruvllm/src/models

Model-architecture implementations optimized for ANE / Metal / CPU. Currently
ships two RuvLTRA models built on Qwen architectures.

## Files
- `mod.rs` - public API + selection guide (size vs perf).
- `ruvltra.rs` - RuvLTRA-Small (Qwen 0.5B, ~500M params); ANE-optimized
  edge inference (~200 tok/s on 38 TOPS ANE).
- `ruvltra_medium.rs` - RuvLTRA-Medium (Qwen2.5-3B, ~3B params); balanced
  quality / performance.

## Related
- `../backends/` host the runtime backends that load these.
- `../models/ruvltra_small.json` (crate root) is the shipped spec.
