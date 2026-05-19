# dragnes / src / routes / api / feedback

`POST /api/feedback` endpoint. Receives clinician/user feedback on a classification (e.g. confirmed label, correction) so the brain backend can learn.

## Important files
- `+server.ts` - SvelteKit `POST` handler. Routes feedback into the brain backend via `$lib/dragnes/brain-client`.

## Related
- Companion endpoints: `../analyze/`, `../similar/`, `../health/`.
