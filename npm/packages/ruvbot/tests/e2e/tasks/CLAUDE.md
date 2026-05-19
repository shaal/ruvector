# ruvbot / tests / e2e / tasks

End-to-end tests for long-running background tasks (workers + queue).

## Files
- `long-running-tasks.test.ts` - Schedules durable jobs through the
  WorkerPool/QueueManager and asserts completion / retry behavior.
