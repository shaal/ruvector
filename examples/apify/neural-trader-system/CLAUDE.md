# apify/neural-trader-system

Self-contained Apify Actor that ships a hand-rolled neural network and trading engine in pure JavaScript (Xavier init, ReLU, simple SGD), without external ML deps. Demonstrates how to package a custom trading model behind the Apify Actor interface.

## Key files
- `src/main.js` - Actor entrypoint. Defines the `NeuralEngine` class and trading pipeline, then drives it through the Apify Actor lifecycle.

## Tech stack
- Node.js, `apify` SDK only (no extra ML dependencies).
- Deployable as an Apify Actor (Docker-based).

## How to run
- `apify push` from this directory to deploy.
- Locally: `node src/main.js` with Apify env vars set (or `apify run`).

## Related
- Sibling `../agentic-synth/` for an LLM-driven Actor.
- Richer trading suite: `examples/neural-trader/`.
