# ui/ruvocal/src/routes/__debug/openai/

Debug endpoint for the OpenAI-compatible model endpoint adapter.

## Files

- `+server.ts` — exposes raw request/response or call inspection against the configured OpenAI-compatible endpoint (`lib/server/endpoints/openai/`). Guard with admin auth before exposing in any non-dev environment.
