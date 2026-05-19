# ui/ruvocal/src/routes/conversation/[id]/

The conversation page (`/conversation/<id>`) — the heart of the chat UI.

## Files

- `+page.svelte` — renders the `ChatWindow` (from `lib/components/chat/`) for this conversation.
- `+page.ts` — universal load: pulls the conversation via `lib/APIClient.ts`.
- `+server.ts` — endpoint actions (e.g. delete / patch the conversation) bound to this id.

## Subdirectories

- `message/[messageId]/prompt/` — endpoint for resending/regenerating from a specific message.
- `share/` — endpoint to create/refresh a shareable link for this conversation.
- `stop-generating/` — endpoint to abort an in-flight generation for this conversation.
