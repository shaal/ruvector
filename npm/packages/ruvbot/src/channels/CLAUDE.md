# ruvbot / src / channels

Multi-channel messaging abstraction (ADR-010). Lets a single bot serve
Slack, Discord, and custom channels through a uniform adapter API.

## Files
- `ChannelRegistry.ts` - Registers and dispatches messages to adapters
  by channel id; surfaces events for inbound/outbound messages.
- `index.ts` - Barrel re-exporting the registry and adapter types.
- `adapters/` - Concrete adapters (Slack, Discord, base class).
