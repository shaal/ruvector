# ui/ruvocal/stub/@reflink/reflink/

Empty stub for the `@reflink/reflink` native package. Resolves to a no-op so that downstream deps (transitively pulling reflink for hard-link / clone optimizations) don't try to build native code in this app's environment.

## Files

- `package.json` — `{ "name": "@reflink/reflink", "version": "0.0.0", "main": "index.js" }`. There's no `index.js` because nothing should actually import it at runtime; the override only satisfies resolution.

Activated by the `overrides` field in `../../../package.json`. Do not remove without auditing transitive deps.
