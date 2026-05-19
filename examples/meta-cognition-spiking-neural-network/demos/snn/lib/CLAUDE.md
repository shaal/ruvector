# demos/snn/lib

JavaScript wrapper around the SIMD-optimized SNN native addon.

## Files
- `SpikingNeuralNetwork.js` - High-level API. Tries to load the native addon (built by node-gyp); exposes `LIFLayer`, network builders (`createFeedforwardSNN`), and encodings (`rateEncoding`, `temporalEncoding`).

## Related
- Native source: `../native/snn_simd.cpp`.
- Build config: `../binding.gyp`.
- Examples: `../examples/`.
