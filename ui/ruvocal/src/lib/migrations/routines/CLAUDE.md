# ui/ruvocal/src/lib/migrations/routines/

Individual MongoDB migration routines, numbered to guarantee ordering. Each routine is a one-shot transformation applied at most once per database.

## Files

- `index.ts` — aggregates and exports all routines in order. **Add new routines here.**
- `01-update-search-assistants.ts` — update assistants for new search shape.
- `02-update-assistants-models.ts` — migrate assistant model references.
- `04-update-message-updates.ts` — reshape stored `message.updates`.
- `05-update-message-files.ts` — migrate file references on messages.
- `06-trim-message-updates.ts` — trim oversized update arrays.
- `08-update-featured-to-review.ts` — rename `featured` to `review` state.
- `09-delete-empty-conversations.ts` (+ `.spec.ts`) — purge empty conversations.
- `10-update-reports-assistantid.ts` — update assistantId field on reports.

## Conventions

- File naming: `NN-short-name.ts`, numeric prefix is the migration ID.
- Routines export `{ _id, name, up(client) }`. Add a `.spec.ts` for non-trivial ones.
