# ruvector-kalshi/examples

Runnable examples exercising the Kalshi integration.

- `bench_signing.rs` — micro-benchmark the RSA-PSS-SHA256 signer (`auth.rs`).
- `list_markets.rs` — call Kalshi REST to enumerate markets.
- `stream_orderbook.rs` — subscribe to a market via WebSocket and print orderbook
  updates.
- `paper_trade.rs` — dry-run end-to-end strategy without sending live orders.
- `live_trade.rs` — same path but with live order submission (gated by runtime
  flag / credentials).
- `validate.rs` — validate credentials / connectivity.
