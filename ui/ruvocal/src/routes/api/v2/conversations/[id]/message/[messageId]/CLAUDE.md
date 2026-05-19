# ui/ruvocal/src/routes/api/v2/conversations/[id]/message/[messageId]/

v2 single-message endpoint within a conversation. Drives the chat-stream / edit / branch flows.

## Files

- `+server.ts` — `GET`/`POST`/`PATCH`/`DELETE` for the message at `[messageId]`. Streams generation updates from `lib/server/textGeneration/` and manipulates the message tree via `lib/utils/tree/`. Tested in `lib/server/api/__tests__/conversations-message.spec.ts`.
