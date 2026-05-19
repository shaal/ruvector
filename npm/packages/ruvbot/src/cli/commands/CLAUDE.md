# ruvbot / src / cli / commands

Individual CLI subcommands exposed by `ruvbot`.

## Files
- `agent.ts` - Agent management (create, list, configure agents).
- `channels.ts` - Manage Slack/Discord/webhook channels.
- `deploy.ts` - Deployment helpers (Docker / Cloud Run flows).
- Additional commands (`start`, `init`, `doctor`, `config`, `memory`,
  `security`, `plugins`, `status`) live alongside as `.ts` files
  registered by `../index.ts`.
