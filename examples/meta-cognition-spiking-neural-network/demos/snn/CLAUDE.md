# demos/snn

State-of-the-art Spiking Neural Network with a SIMD-accelerated C++ N-API native addon plus JS wrappers.

## Files
- `package.json` - `snn-simd` v1.0.0. Scripts: `install`/`build` -> `node-gyp rebuild`, `test` -> `examples/pattern-recognition.js`, `benchmark` -> `examples/benchmark.js`. Deps: `node-addon-api ^7`, dev: `node-gyp ^10`. `gypfile: true`.
- `binding.gyp` - node-gyp build config for the C++ native addon.

## Subdirectories
- `lib/SpikingNeuralNetwork.js` - High-level JS wrapper around the native addon.
- `native/snn_simd.cpp` - C++ N-API implementation: LIF neurons, STDP learning, SIMD membrane updates, lateral inhibition, homeostatic plasticity.
- `examples/` - Pattern-recognition demo and SNN-vs-JS benchmark.

## Build / run
```
npm install            # builds native addon via node-gyp
npm test               # runs examples/pattern-recognition.js
npm run benchmark      # examples/benchmark.js
```

## Tech stack
- Node >=16, node-addon-api 7, node-gyp 10, C++ with SIMD intrinsics.

## Related
- Parent: `../CLAUDE.md`.
- Guide: `../../docs/SNN-GUIDE.md`.
