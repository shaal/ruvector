# dragnes / src / lib / dragnes

Core DrAgnes module - browser-side dermatology pipeline. Loads MobileNetV3 Small WASM (from `@ruvector/cnn`) for skin-lesion classification, computes ABCDE dermoscopic scores, queries the brain backend for similar cases, and handles privacy / offline / federation concerns.

## Important files
- `index.ts` - barrel export (public API of the module).
- `classifier.ts` - `DermClassifier` class: loads the WASM CNN, runs inference, generates Grad-CAM heatmaps; demo fallback when WASM is unavailable.
- `abcde.ts` - ABCDE dermoscopic scoring (Asymmetry, Border, Color, Diameter, Evolving).
- `brain-client.ts` - HTTP client for similar-case + literature lookups; backs the `/api/similar` and `/api/analyze` endpoints.
- `config.ts` - module-level configuration (mirrors `../../../dragnes.config.ts`).
- `datasets.ts`, `ham10000-knowledge.ts` - dataset-specific knowledge (HAM10000 labels, prior stats).
- `preprocessing.ts` - image preprocessing pipeline (tensor conversion, normalization).
- `privacy.ts`, `witness.ts` - differential privacy / witness-based anonymity guards.
- `federated.ts` - federated learning client.
- `offline-queue.ts` - IndexedDB-backed offline submission queue.
- `benchmark.ts` - in-browser benchmark of the classifier.
- `deployment-runbook.ts` - deployment runbook embedded as typed config.
- `types.ts` - shared types (`LesionClass`, `ClassificationResult`, `GradCamResult`, `ImageTensor`, ...).

## Related
- UI components consuming this module: `../components/`. HTTP endpoints in `../../routes/api/`.
