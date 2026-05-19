# @ruvector/spiking-neural

High-performance Spiking Neural Network (SNN) with SIMD optimization — CLI and Node SDK. Uses LIF neurons and STDP learning; ships C++ native code via `node-gyp` for SIMD acceleration.

## Key files

- `package.json` — `@ruvector/spiking-neural` v1.0.1; main `src/index.js`; bins `spiking-neural` / `snn` -> `./bin/cli.js`.
- `binding.gyp` — `node-gyp` build config for the native add-on.
- `LICENSE`.

## Subdirectories

- `bin/cli.js` — published CLI; also wired to `npm test`, `npm run benchmark`, `npm run demo`.
- `examples/` — `basic.js`, `benchmark.js`, `pattern-recognition.js`.
- `native/` — `snn_simd.cpp` (SIMD C++ source compiled by `node-gyp`).

Note: `src/` is referenced via `main: "src/index.js"` and listed in `files` but not present in this checkout.

## Scripts

- `test` -> `node bin/cli.js test`
- `benchmark` -> `node bin/cli.js benchmark`
- `demo` -> `node bin/cli.js demo pattern`
- `build:native` -> `node-gyp rebuild`
- `prepublishOnly` -> `npm test`

## Deps

- Build-only: `node-gyp`, `node-addon-api`.

## Related

- Pure C++ + JS package; no Rust crate dependency.
