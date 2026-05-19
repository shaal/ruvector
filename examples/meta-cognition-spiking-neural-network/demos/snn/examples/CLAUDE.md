# demos/snn/examples

Runnable SNN demonstrations exercising the native addon.

## Files
- `pattern-recognition.js` - Demonstrates rate-coded input encoding, STDP learning, pattern classification, and lateral inhibition (winner-take-all). Runs as `npm test`.
- `benchmark.js` - Measures SIMD-vs-JS performance for the LIF layer / network forward pass. Runs as `npm run benchmark`.

## Related
- Parent: `../CLAUDE.md`.
- Native code: `../native/snn_simd.cpp`.
- JS wrapper: `../lib/SpikingNeuralNetwork.js`.
