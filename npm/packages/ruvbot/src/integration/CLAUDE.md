# ruvbot / src / integration

Integration bounded context (ADR-005). Hosts external system adapters:
LLM providers, Slack, and HTTP webhooks. Each subdir publishes its own
barrel reachable through the package's subpath exports
(`ruvbot/integrations/slack`, `ruvbot/integrations/webhooks`).

## Files
- `index.ts` - Barrel re-exporting the providers / slack / webhooks
  modules.

## Subdirectories
- `providers/` - LLM provider implementations.
- `slack/` - Slack bolt integration.
- `webhooks/` - Inbound/outbound webhook handlers.
