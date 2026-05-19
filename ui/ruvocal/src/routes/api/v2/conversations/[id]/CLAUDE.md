# ui/ruvocal/src/routes/api/v2/conversations/[id]/

v2 single-conversation endpoint.

## Files

- `+server.ts` — `GET`/`PATCH`/`DELETE` for the conversation `[id]`. Uses `resolveConversation` + `requireAuth`.

## Subdirectories

- `message/` — nested per-message subroutes.
