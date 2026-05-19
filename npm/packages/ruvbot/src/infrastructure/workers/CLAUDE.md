# ruvbot / src / infrastructure / workers

Background worker pool abstraction (ADR-004). Wraps the optional
`bullmq` queue with a strongly-typed worker registration API used for
embedding generation, pattern learning, and other async tasks.

## Files
- `index.ts` - Exports `WorkerPool` and related types re-exported by
  the package root.
