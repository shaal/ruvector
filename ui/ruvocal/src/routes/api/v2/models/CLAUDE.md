# ui/ruvocal/src/routes/api/v2/models/

v2 model catalog API. Models are addressed as `[namespace]/[model]` (e.g. `huggingface/llama-3.1-8b`).

## Files

- `+server.ts` — `GET` list of models visible to the caller.

## Subdirectories

- `[namespace]/` — namespace-scoped routes including `[model]/` and `subscribe/`.
- `old/` — legacy/back-compat models view.
- `refresh/` — endpoint that re-pulls the model catalog from upstream router.
