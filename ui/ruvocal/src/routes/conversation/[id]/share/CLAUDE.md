# ui/ruvocal/src/routes/conversation/[id]/share/

Create / refresh a shareable link for this conversation.

## Files

- `+server.ts` — `POST` creates (or rotates) a `SharedConversation` record, returning the public share id. The short URL is served via `routes/r/[id]/`. Client uses `lib/createShareLink.ts` + `lib/components/ShareConversationModal.svelte`.
