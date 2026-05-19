# @ruvector/ruvllm

Self-learning LLM runtime — TurboQuant KV-cache (6-8x compression), SONA adaptive learning, FlashAttention, speculative decoding, GGUF inference. Dual CJS/ESM TS package fronting native NAPI bindings.

## Key files

- `package.json` — `@ruvector/ruvllm` v2.5.5; main `dist/cjs/index.js`, module `dist/esm/index.js`; bin `ruvllm` -> `./bin/cli.js`.
- `tsconfig.json` / `tsconfig.esm.json` — dual-build configs.
- `Dockerfile.benchmark`, `Dockerfile.test` — containers for benchmarking/CI.

## Subdirectories

- `bin/` — `cli.js` (prebuilt `ruvllm` CLI).
- `npm/` — per-platform NAPI sub-packages (darwin-arm64/x64, linux-arm64/x64-gnu, win32-x64-msvc).
- `scripts/` — model-comparison harnesses, `huggingface/publish.sh`, `training/` datasets and generators.
- `src/` — TypeScript source (with checked-in `.js`/`.d.ts`).
- `src/benchmarks/` — embedding/model-comparison/routing benchmark sources.
- `test/` — Node `--test` suites (`basic`, `features`, `advanced-features`) plus `benchmark.js`.

## Published API

- `.` -> main runtime entry (engine/session/lora/sona/training/streaming/etc.)
- `./simd` -> SIMD helpers (`simd.{ts,d.ts,js}`)

## Scripts

- `build` -> `build:cjs && build:esm`
- `build:cjs` -> `tsc`; `build:esm` -> `tsc -p tsconfig.esm.json`
- `build:native` / `build:debug` -> `napi build -p ruvllm --manifest-path ../../../examples/ruvLLM/Cargo.toml -F napi`
- `artifacts`, `universal`, `version` -> NAPI helpers
- `test` -> `node --test test/*.test.js`
- `typecheck`, `clean`

## Key deps

`chalk`, `commander`, `ora`. NAPI binaries via optional deps `@ruvector/ruvllm-<platform>`.

## Related

- Rust crate / source: `examples/ruvLLM/` (per `homepage` and `build:native` manifest path).
- Sibling: `npm/packages/ruvllm-cli/` — standalone, lighter CLI for the same runtime.
- Sibling: `npm/packages/ruvllm-linux-arm64-gnu/` and other platform subpackages here.
