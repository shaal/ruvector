# ui/ruvocal/src/routes/admin/stats/compute/

Admin endpoint that triggers recomputation of conversation statistics.

## Files

- `+server.ts` — `POST` handler; runs `lib/jobs/refresh-conversation-stats.ts`. Admin-token gated.
