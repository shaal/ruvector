# ui/ruvocal/src/routes/login/callback/

OpenID Connect callback. Exchanges the authorization code for tokens, resolves the user, sets the session cookie, then redirects to the originally requested URL.

## Files

- `+server.ts` — `GET /login/callback` handler.
- `updateUser.ts` (+ `updateUser.spec.ts`) — pure helper that upserts the user record from OIDC claims (used by the handler and tested standalone).
