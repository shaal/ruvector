# ui/ruvocal/src/routes/api/user/validate-token/

Token validation endpoint.

## Files

- `+server.ts` — `POST { token }` validates an API token via `lib/server/apiToken.ts` and returns the resolved user (or 401). Useful for CLI / SDK clients to verify credentials.
