# docs/research/claude-code-rvsource/extracted/

Machine-extracted artifacts from the claude-code v2.1 binary, organized by domain. Not hand-written prose - these files are inputs/outputs of the extraction pipeline.

## Contents

- `metrics.json` - extraction metrics summary.
- `rvf/` - per-domain RVF bundles (config, core, permissions, telemetry, tools, types, ui, uncategorized) plus a master.rvf, each with a `.manifest.json` sidecar.
- `source/` - extracted JS source organized by domain plus a `witness.json` attesting to the extraction.

## Related

- `../` - prose analysis docs.
- `../versions/v2.1.x/` - alternate per-feature tree view.
