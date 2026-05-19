# apify/agentic-synth/src

Source directory for the agentic-synth Apify Actor.

## Files
- `main.js.backup` - Apify Actor entrypoint (kept as a `.backup` while paused). Imports `apify`, `@google/generative-ai`, integrations, embeddings helpers, and the shared `memory-persistence` module; conditionally loads native `@ruvector/ruvllm` for TRM/SONA self-learning generation.

## Notes
- Expects sibling modules `integrations.js` and `embeddings.js` (not yet present in this checkout).
- Charging is wrapped in a safe helper that swallows errors when monetization is not configured.
- Rename `main.js.backup` to `main.js` to make the Actor runnable.

## Related
- Parent: `../CLAUDE.md`.
- Sibling Actor source: `../../neural-trader-system/src/`.
