# demos/snn/native

C++ N-API source for the SIMD-optimized Spiking Neural Network addon.

## Files
- `snn_simd.cpp` - Full implementation: Leaky Integrate-and-Fire (LIF) neurons, STDP plasticity, SIMD-accelerated membrane potential updates, lateral inhibition, homeostatic plasticity. Exposed to Node via N-API.

## Build
Built by node-gyp through `../binding.gyp` when running `npm install`/`npm run build` in `../`.

## Related
- JS wrapper: `../lib/SpikingNeuralNetwork.js`.
- Examples: `../examples/`.
