# ruvector-kalshi

Kalshi prediction-market exchange integration for the RuVector Neural Trader
(ADR-153). Provides RSA-PSS-SHA256 request signing, typed REST/WebSocket DTOs,
normalization into `neural_trader_core::MarketEvent` so downstream coherence,
attention, and replay pipelines work unchanged, plus secret loading from GCP
Secret Manager / local PEM.

## Layout

- `Cargo.toml` — `publish = false`. Deps: `neural-trader-*` crates, reqwest
  (rustls), tokio-tungstenite (WS, rustls + webpki roots), rsa (sha2), sha2,
  base64, chrono, anyhow.
- `src/lib.rs` — module roots, constants `KALSHI_VENUE_ID = 1001`,
  `KALSHI_API_URL`, `KALSHI_WS_URL`, `KALSHI_PRICE_FP_SCALE = 1_000_000`, and
  `KalshiError`.
- `src/auth.rs` — RSA-PSS-SHA256 signer.
- `src/secrets.rs` — load PEM from GCP Secret Manager or disk.
- `src/rest.rs` — REST client scaffold (live calls gated by runtime flag).
- `src/ws.rs`, `src/ws_client.rs` — WebSocket live transport.
- `src/rate_limit.rs` — Kalshi-specific rate limiting.
- `src/models.rs` — typed market/event/order/fill DTOs.
- `src/normalize.rs` — Kalshi payload -> `MarketEvent` normalization.
- `src/strategy_adapter.rs` — adapt neural-trader strategies to Kalshi semantics.
- `src/brain.rs` — Shared-Brain integration for trade signals.
- `examples/` — `bench_signing.rs`, `list_markets.rs`, `live_trade.rs`,
  `paper_trade.rs`, `stream_orderbook.rs`, `validate.rs`.
- `tests/live_smoke.rs` — live smoke test (requires credentials).

## Related

- `crates/neural-trader-core`, `crates/neural-trader-coherence`,
  `crates/neural-trader-replay`, `crates/neural-trader-strategies`.
