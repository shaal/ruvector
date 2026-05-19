# src/

TypeScript source for `@ruvector/ruvllm`. Built twice via `tsc` and `tsc -p tsconfig.esm.json` into `../dist/cjs/` and `../dist/esm/`. Each `.ts` ships alongside its compiled `.js`/`.d.ts` here as well.

- `index.ts` — main barrel; exports engine, session, models, training, lora, sona, contrastive, federated, intelligence, streaming, export, simd, native, and types.
- `engine.ts` — inference engine entry.
- `session.ts` — conversational sessions / context handling.
- `models.ts` — model registry + loading.
- `native.ts` — NAPI loader for the platform-specific `.node` binaries.
- `simd.ts` — SIMD helpers (also published under `./simd` subpath).
- `streaming.ts` — token streaming.
- `export.ts` — export model artifacts.
- `lora.ts` — LoRA adapter integration.
- `sona.ts` — SONA adaptive-learning integration.
- `contrastive.ts` — contrastive training/inference.
- `federated.ts` — federated learning protocol.
- `intelligence.ts` — intelligence/meta layer.
- `training.ts` — training driver.
- `types.ts` — shared type definitions.

## Subdirectory

- `benchmarks/` — embedding / model-comparison / routing benchmark sources.
