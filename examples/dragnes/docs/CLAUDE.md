# dragnes / docs

Product, architecture, and compliance docs for the DrAgnes platform.

## Important files
- `architecture.md` - system architecture (SvelteKit, WASM CNN, brain sync, offline queue).
- `HAM10000_analysis.md`, `HAM10000_stats.json` - analysis + descriptive statistics of the HAM10000 dermoscopy dataset (the model's training corpus).
- `data-sources.md` - upstream datasets and licensing.
- `dermlite-integration.md` - integration notes for the DermLite dermatoscope.
- `competitive-analysis.md` - market positioning.
- `deployment.md` - deployment guide (companion to `../Dockerfile` and `../cloud-run.yaml`).
- `future-vision.md` - roadmap / north-star.
- `hipaa-compliance.md` - HIPAA-aligned controls.

## Related
- App code: `../src/`. Deploy assets: `../Dockerfile`, `../cloud-run.yaml`, `../scripts/deploy.sh`.
