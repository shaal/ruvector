# ui/ruvocal/src/routes/api/v2/models/old/

Backward-compatible model listing in the legacy shape.

## Files

- `+server.ts` — `GET` returns the model catalog using the legacy schema (flat array without namespace). Kept to avoid breaking older SDK / API consumers; prefer `../+server.ts` for new code.
