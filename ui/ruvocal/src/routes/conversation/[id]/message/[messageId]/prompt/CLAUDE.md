# ui/ruvocal/src/routes/conversation/[id]/message/[messageId]/prompt/

Endpoint to re-prompt / regenerate from a specific message — creates a new sibling response branch using the tree helpers.

## Files

- `+server.ts` — `POST` triggers a new generation starting from `[messageId]`. Adds a sibling via `lib/utils/tree/addSibling.ts`, streams updates from `lib/server/textGeneration/`, and registers the generation with `lib/server/abortRegistry.ts` so it can be cancelled via `../../../stop-generating/`.
