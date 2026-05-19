# apify/neural-trader-system/src

Source directory for the neural-trader-system Apify Actor.

## Files
- `main.js` - Actor entrypoint. Implements `NeuralEngine` (configurable layers/neurons/activation/dropout/learning rate, Xavier init, ReLU activation) and the trading orchestration around it, wrapped in the Apify Actor lifecycle.

## Notes
- Pure JS, no external ML dependencies beyond `apify`.
- Hardcoded 50-feature input vector.

## Related
- Parent: `../CLAUDE.md`.
- Sibling Actor: `../../agentic-synth/src/`.
