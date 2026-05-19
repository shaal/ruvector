# ui/ruvocal/src/lib/server/files/

Server-side file handling for chat attachments.

## Files

- `uploadFile.ts` — accepts an uploaded file, validates MIME (`lib/constants/mime.ts`), stores it (likely GridFS / object store), returns a file ref attached to a message.
- `downloadFile.ts` — fetches a stored file by id and streams it back to the client (used by message renderers and OG-image generation).
