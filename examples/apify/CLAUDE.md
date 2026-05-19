# examples/apify

Apify Actor integrations of RuVector. Each subdirectory is a separate Apify Actor: a JavaScript program packaged to run on the Apify platform that wires RuVector / AgentDB / RuvLLM components into a hosted workflow.

## Subdirectories
- `agentic-synth/` - Agentic synthesis Actor combining Google Generative AI, Apify scraper integrations, RuvLLM (TRM/SONA self-learning), and shared memory persistence.
- `neural-trader-system/` - Standalone neural-trading Actor implementing a classifier/trading engine from scratch in JS for Apify.

## Tech stack
- JavaScript (ES modules), `apify` SDK, `@google/generative-ai`, optional native `@ruvector/ruvllm` add-on.
- Runs on the Apify Actor platform (Docker-based serverless functions).

## Related examples
- `../neural-trader/` for the full Node.js neural-trader integration outside of Apify.
- `../meta-cognition-spiking-neural-network/` for the SNN/attention/SIMD demos that share the AgentDB stack.
