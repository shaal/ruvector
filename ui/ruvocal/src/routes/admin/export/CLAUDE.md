# ui/ruvocal/src/routes/admin/export/

Admin endpoint that streams a full data export.

## Files

- `+server.ts` — `GET` handler that produces a zipped (yazl) / parquet export of conversations and related data. Admin-token gated.
