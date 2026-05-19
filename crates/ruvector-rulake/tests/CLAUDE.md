# ruvector-rulake/tests

Integration tests / M1 acceptance gates.

## Files

- `federation_smoke.rs` — verifies (1) `RuLake::search_one` matches direct `RabitqPlusIndex::search` top-k (modulo tie order), (2) backend mutation bumps the cache generation and triggers re-prime, (3) federated search across two backends returns globally-correct top-k, (4) cache-hit path is measurably faster than cache-miss.
