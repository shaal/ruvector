# src/decompiler/

Self-decompiler engine — pure JavaScript modules (no TS sources here). Shipped via `files: ["src/decompiler/"]` so users get them at runtime.

- `index.js` — decompiler entry point.
- `model-decompiler.js`, `module-splitter.js`, `module-tree.js`, `reconstructor.js` — core decompilation pipeline.
- `api-prober.js`, `name-predictor.js`, `reference-tracker.js`, `statement-parser.js`, `style-improver.js`, `validator.js`, `subcategories.js` — supporting passes.
- `npm-fetch.js` — fetches package metadata from npm.
- `metrics.js`, `witness.js` — measurement / verification helpers.
