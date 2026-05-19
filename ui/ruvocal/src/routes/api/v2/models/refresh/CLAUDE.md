# ui/ruvocal/src/routes/api/v2/models/refresh/

Forces a re-pull of the model catalog from upstream.

## Files

- `+server.ts` — `POST` triggers `lib/server/models.ts` to refresh from the configured router / `OPENAI_BASE_URL`. Useful when new models appear without restarting the app. Usually admin-token gated.

## Related

- Skill `/add-model-descriptions` populates `chart/env/dev.yaml` and `prod.yaml` with descriptions after a refresh.
