# rvf/dashboard/src

TypeScript source for the RVF dashboard.

## Top-level files

- `main.ts` - App bootstrap and view routing.
- `api.ts` - REST API client to the RVF server.
- `solver.ts` - Wrapper around the `@ruvector/rvf-solver` WASM.
- `ws.ts` - WebSocket client for live telemetry.

## Subdirs

- `views/` - High-level page views (Atlas, Boundaries, Coherence, Discovery, Docs, Download, Dyson, Life, Memory, Planet, Solver, Status, Witness, BlindTest).
- `three/` - Three.js scenes (AtlasGraph, CausalFlow, CoherenceSurface, DysonSphere3D, OrbitPreview, PlanetSystem3D, LODController).
- `charts/` - D3-based charts (LightCurve, MoleculeMatrix, Radar, Spectrum).
- `components/` - Small reusable UI pieces (Sidebar, TimeScrubber, DownloadProgress, WitnessLog).
- `styles/main.css` - App-wide styles.

## Related

- Parent: `../`.
- Built bundle: `../dist/`.
