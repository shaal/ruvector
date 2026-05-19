# ui/ruvocal/src/routes/conversation/[id]/stop-generating/

Aborts any in-flight generation for this conversation.

## Files

- `+server.ts` — `POST` looks up the active generation in `lib/server/abortRegistry.ts` and signals abort. The streaming endpoint records the abort via `lib/server/abortedGenerations.ts`. Tested in `lib/server/__tests__/conversation-stop-generating.spec.ts`. Client trigger: `lib/components/StopGeneratingBtn.svelte` + `lib/stores/isAborted.ts`.
