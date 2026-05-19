# ruvbot / src / channels / adapters

Concrete channel adapters used by `ChannelRegistry`.

## Files
- `BaseAdapter.ts` - Abstract class. Defines lifecycle (`connect`,
  `disconnect`, `sendMessage`) and event hooks shared across adapters.
- `SlackAdapter.ts` - Slack adapter built on `@slack/bolt` / `@slack/
  web-api` (optional deps).
- `DiscordAdapter.ts` - Discord adapter for Discord gateway / REST.
