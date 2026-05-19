# ui/ruvocal/src/routes/conversation/

The conversation-page surface plus a top-level conversation creation endpoint.

## Files

- `+server.ts` — `POST` creates a new conversation, returns the id (used to navigate to `/conversation/[id]`).

## Subdirectories

- `[id]/` — the conversation page itself and its server actions (`message`, `share`, `stop-generating`).
