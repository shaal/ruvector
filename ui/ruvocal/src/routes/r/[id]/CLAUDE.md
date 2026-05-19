# ui/ruvocal/src/routes/r/[id]/

Short URL `/r/<id>` for shared conversations — resolves the share record and redirects to the canonical conversation view.

## Files

- `+page.ts` — universal load: looks up the share by `[id]` and triggers a `redirect` to the read-only conversation render or the import flow (`/api/v2/conversations/import-share`). Counterpart of `routes/conversation/[id]/share/+server.ts` + `lib/createShareLink.ts`.
