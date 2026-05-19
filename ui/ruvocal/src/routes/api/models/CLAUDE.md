# ui/ruvocal/src/routes/api/models/

Legacy model catalog endpoint.

## Files

- `+server.ts` — `GET` returns the list of configured models from `lib/server/models.ts` (filtered by user capability / pro status).

The v2 surface (`../v2/models/`) is more expressive (namespaced, subscribe streams).
