# apify/agentic-synth

Apify Actor that synthesizes agentic outputs by combining Apify scraper data with Google Generative AI, embedding generation, and (optionally) RuvLLM-powered TRM/SONA self-learning. Charges per Apify monetization event.

## Key files
- `src/main.js.backup` - Actor entrypoint (currently kept as `.backup`). Loads the Apify SDK, Gemini client, RuvLLM native binding via `createRequire`, and shared memory persistence; orchestrates synthesis.
- (Sibling expected modules referenced from main: `integrations.js`, `embeddings.js`, `../../../shared/memory-persistence.js`.)

## Tech stack
- Node.js ES modules.
- `apify`, `@google/generative-ai`.
- Optional native module `@ruvector/ruvllm` for self-learning generation.
- Apify Actor monetization (`Actor.charge`).

## How to run
- Deploy via `apify push` from this directory once a `package.json` / `Dockerfile` is restored.
- Locally: `node src/main.js` (rename from `.backup`) with Apify env vars set.

## Related
- Sibling `../neural-trader-system/` for another Apify Actor.
- `examples/neural-trader/` for the broader Node toolkit this Actor wraps.
