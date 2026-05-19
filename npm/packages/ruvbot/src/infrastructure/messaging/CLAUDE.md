# ruvbot / src / infrastructure / messaging

Event bus / queue abstraction (ADR-004). Domain events are dispatched
through an in-process `EventBus`; durable jobs go through a
`QueueManager` implemented on top of the optional `bullmq` + `ioredis`
stack.

## Files
- `index.ts` - Exports `EventBus`, `DomainEvent`, `EventHandler`,
  `Subscription`, `QueueManager`, `Job`, `JobOptions`. These types are
  also re-exported from the package root.
