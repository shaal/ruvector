# docs/research/claude-code-rvsource/extracted/rvf/

Per-domain RVF (ruvector format) bundles extracted from claude-code v2.1. Each `*.rvf` has a `*.rvf.manifest.json` sidecar describing its contents.

## Bundles

- `master.rvf` - master bundle aggregating all domains.
- `config.rvf` - configuration domain.
- `core.rvf` - agent loop, context, session, streaming.
- `permissions.rvf` - permission system.
- `telemetry.rvf` - telemetry events / OTEL.
- `tools.rvf` - tool dispatch and MCP client.
- `types.rvf` - API endpoints and class hierarchy.
- `ui.rvf` - command definitions / UI.
- `uncategorized.rvf` - leftover unclassified modules.

## Related

- `../source/` - extracted JS source (one folder per domain).
- `../../14-source-extraction.md` - extraction methodology.
