# ruvbot / src / integration / webhooks

HTTP webhook adapter. Lets ruvbot send outbound webhooks on events
and receive inbound webhook calls for chat / command triggers.
Exposed via the package subpath export `ruvbot/integrations/webhooks`.

## Files
- `index.ts` - Exports webhook handlers, signing helpers, and config
  types.
