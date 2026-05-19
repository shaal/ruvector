# ui/ruvocal/stub/

Local npm overrides (stub packages) used to short-circuit native dependencies that can't (or shouldn't) be installed in the chat-ui build.

## Subdirectories

- `@reflink/` — namespace containing the `@reflink/reflink` stub.

Wired up via the `overrides` field in `../package.json` (`"@reflink/reflink": "file:stub/@reflink/reflink"`).
