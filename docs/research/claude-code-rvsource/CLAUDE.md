# docs/research/claude-code-rvsource/

Source archeology / decompilation of **claude-code v2.1** packaged into ruvector's RVF format. Contains both prose analysis (numbered docs) and large machine-extracted artifacts under `extracted/` and `versions/`. Audience: researchers studying how claude-code is structured for the ruvector-integration design.

## Prose docs (read in order)

- `00-index.md` - section index.
- `01-overview-and-binary-structure.md`, `02-tool-system.md`, `03-agent-loop-and-execution-flow.md`.
- `04-permission-system.md`, `05-mcp-integration.md`, `06-hooks-system.md`.
- `07-context-and-session-management.md`, `08-configuration-and-environment.md`.
- `09-agent-and-subagent-system.md`, `10-models-and-api.md`.
- `11-telemetry-and-observability.md`, `12-dependency-graph.md`, `13-extension-points.md`.
- `14-source-extraction.md`, `15-core-module-analysis.md`, `16-call-graphs.md`.
- `17-class-hierarchy.md`, `18-state-machines.md`.
- `19-ruvector-integration-guide.md` - integration design.
- `20-sota-decompiler-research.md`, `21-model-weight-analysis.md`.

## RVF artifacts

- `claude-code-v2.1-runnable.rvf` + `.manifest.json` - runnable RVF bundle.

## Subdirs

- `extracted/` - extracted JS sources organized by domain (`source/{config,core,permissions,telemetry,tools,types,ui,uncategorized}`) plus per-domain `.rvf` files in `extracted/rvf/`.
- `versions/v2.1.x/tree/` - per-feature directory tree of extracted modules. Names are derived from interesting symbols (e.g. `asyncgenerator/`, `bedrockclient/`, `select-pane/`).

## Related

- `../../adr/` - ruvector ADRs that may cite this archeology.
- `../rvf/` - RVF format spec.
