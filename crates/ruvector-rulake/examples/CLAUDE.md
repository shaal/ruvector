# ruvector-rulake/examples

Runnable examples (`cargo run --example <name> -p ruvector-rulake`).

## Files

- `sidecar_daemon.rs` — minimal cache-sidecar daemon implementing the ADR-155 bundle protocol. Watches a publish directory for `table.rulake.json` updates and calls `RuLake::refresh_from_bundle_dir` to keep reader caches coherent with publisher witnesses.
- `warm_restart.rs` — three-phase end-to-end demo: PUBLISHER primes cache + writes bundle, READER warm-restarts from disk in ms, COLD reader pays full backend prime cost. Summary reports warm-vs-cold speedup.
