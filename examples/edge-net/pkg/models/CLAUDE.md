# edge-net/pkg/models

JS-side model layer: loading, optimization, distribution, integrity, training utilities, and a registry.

## Important files
- `model-loader.js` / `loader.js` — model loading entry points.
- `model-registry.js` + `registry.json` — registered model metadata.
- `model-optimizer.js` — quantization / size optimization.
- `distribution.js` — model distribution over the network.
- `integrity.js` — signature/checksum verification.
- `adapter-hub.js`, `adapter-security.js` — LoRA-style adapter management.
- `microlora.js` — MicroLoRA helpers (pairs with `../../src/ai/lora.rs`).
- `training-utils.js` — training helpers.
- `wasm-core.js` — WASM model runtime glue.
- `models-cli.js` — CLI for managing models.
- `benchmark.js` — model-level benchmarks.
- `model-utils.js` — shared utilities.

## Related
- Architecture: `../../docs/architecture/MODEL_OPTIMIZATION_DISTRIBUTION.md`.
