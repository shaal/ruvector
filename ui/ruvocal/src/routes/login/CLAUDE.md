# ui/ruvocal/src/routes/login/

OpenID Connect login entrypoint.

## Files

- `+server.ts` — `GET /login` builds the OIDC authorize URL via `lib/server/auth.ts` (`openid-client`) and redirects the user there.

## Subdirectories

- `callback/` — OIDC redirect target; exchanges the code for tokens and resolves/creates the user.

## Related

- Logout: `../logout/`.
- Docs: `docs/source/configuration/open-id.md`.
