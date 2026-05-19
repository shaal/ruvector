# ruvector-kalshi/src

Source for the Kalshi exchange integration (ADR-153).

- `lib.rs` — module roots, constants (`KALSHI_VENUE_ID`, REST/WS URLs,
  `KALSHI_PRICE_FP_SCALE`), `KalshiError`.
- `auth.rs` — RSA-PSS-SHA256 signer (Kalshi's required signing scheme).
- `secrets.rs` — load private key PEM from GCP Secret Manager or local file.
- `rest.rs` — REST client scaffold (HTTPS via reqwest+rustls).
- `ws.rs`, `ws_client.rs` — WebSocket live transport via tokio-tungstenite.
- `rate_limit.rs` — Kalshi-specific rate limiting.
- `models.rs` — typed DTOs for market / event / order / fill payloads.
- `normalize.rs` — convert Kalshi payloads into `neural_trader_core::MarketEvent`
  (price upscaled by `KALSHI_PRICE_FP_SCALE`).
- `strategy_adapter.rs` — wire neural-trader strategies into Kalshi order flow.
- `brain.rs` — integration with the Shared Brain for trade signal sharing.
