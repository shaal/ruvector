# agentic-synth/src/routing

Model routing for `@ruvector/agentic-synth` — picks the best provider/model for a generation request.

## Files

- `index.ts` — public exports.
- `model-router.js` — `ModelRouter` that selects between Gemini and OpenRouter based on `SynthConfig` and task hints.
