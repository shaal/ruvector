# ui/ruvocal/src/lib/server/endpoints/

Model-endpoint adapters: the layer that takes a `Model` config and produces a streaming text/image/document response.

## Files

- `endpoints.ts` — endpoint dispatcher; given a model, returns the configured endpoint implementation.
- `preprocessMessages.ts` — pre-processes the message list (file injection, role mapping, system prompt assembly) before sending to a provider.
- `images.ts` — image-generation endpoint adapter.
- `document.ts` — document-handling endpoint (parsing/embedding for RAG-style use).

## Subdirectories

- `openai/` — OpenAI-compatible chat/completion adapter (default).
