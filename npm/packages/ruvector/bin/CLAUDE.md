# bin/

Published executables for the `ruvector` package.

- `cli.js` — large prebuilt CLI bundle; mapped to the `ruvector` bin in `package.json`. Exposes hooks, vector ops, intelligence engine, decompiler, optimizer commands.
- `mcp-server.js` — prebuilt MCP (Model Context Protocol) server bundle that exposes RuVector tools to MCP clients.

Both files are checked into the tree and shipped via the `files: ["bin/"]` entry.
