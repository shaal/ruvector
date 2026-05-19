# ui/ruvocal/src/routes/api/v2/conversations/import-share/

v2 endpoint to import a shared conversation into the calling user's account.

## Files

- `+server.ts` — `POST { shareId }` clones the shared conversation (resolved via the `SharedConversation` collection) into a new `Conversation` owned by the caller. Counterpart to the share endpoint at `routes/conversation/[id]/share/+server.ts` and the client helper `lib/createShareLink.ts`.
