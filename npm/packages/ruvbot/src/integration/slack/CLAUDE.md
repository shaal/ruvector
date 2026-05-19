# ruvbot / src / integration / slack

Slack bot integration built on `@slack/bolt` and `@slack/web-api`
(optional deps). Reachable via the package subpath export
`ruvbot/integrations/slack`.

## Files
- `index.ts` - Exports the Slack app factory, event handlers, and
  configuration types consumed by `ChannelRegistry`.
