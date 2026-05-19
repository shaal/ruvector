# ui/ruvocal/src/routes/api/conversation/[id]/message/[messageId]/

Legacy single-message endpoint within a conversation.

## Files

- `+server.ts` — `GET`/`PATCH`/`DELETE` for the message at `[messageId]` inside conversation `[id]`. Uses `lib/utils/tree/` to manipulate the message tree.
