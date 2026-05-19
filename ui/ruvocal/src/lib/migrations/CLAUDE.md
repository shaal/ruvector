# ui/ruvocal/src/lib/migrations/

MongoDB migration framework. Migrations are versioned and run once per database; status is tracked in a `migrations` collection with the `lock` helper preventing concurrent runs.

## Files

- `migrations.ts` — migration runner: discovers `routines/index.ts`, applies pending migrations in order, records results.
- `lock.ts` — distributed lock to ensure only one process applies migrations at a time.
- `migrations.spec.ts` — runner tests.

## Subdirectories

- `routines/` — individual migration files (`NN-description.ts`), aggregated through `routines/index.ts`.

## Conventions

- Add new migrations as `routines/NN-name.ts` and append to `routines/index.ts`.
- Each routine exports `{ _id, name, up }` (and optionally a tests file).
