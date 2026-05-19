# neural-trader-strategies

Venue-agnostic strategy + risk-gate runtime for the RuVector Neural Trader (ADR-153). Defines the canonical `Intent`, mandatory
`RiskGate`, and concrete strategies (EV-Kelly, attention scalper, coherence arb) that consume normalized `MarketEvent`s from
`neural-trader-core` and emit at most one `Intent` per event.

## Files

- `Cargo.toml` — research-tier crate (`publish = false`). Depends on `neural-trader-core`, `neural-trader-coherence`,
  `ruvector-attention` (default-features off), serde, anyhow, thiserror. Lints heavily relaxed but `correctness`/`suspicious` denied.
- `src/lib.rs` — declares `Strategy` trait, re-exports all strategy / risk / intent types.

## Public API surface

Re-exported from `lib.rs`:
- `Strategy` (trait) — implementer interface, returns `Option<Intent>`.
- `intent::{Action, Intent, Side}` — venue-agnostic order intent.
- `risk::{RiskGate, RiskConfig, RiskDecision, RejectReason, PortfolioState, Position}` — mandatory risk wrapper enforcing
  position cap, daily-loss kill, concentration, min-edge, live-trade env flag.
- `ev_kelly::{ExpectedValueKelly, ExpectedValueKellyConfig}` — first concrete strategy.
- `attention_scalper::{AttentionScalper, AttentionScalperConfig}`.
- `coherence_arb::{CoherenceArb, CoherenceArbConfig}`.
- `coherence_bridge::{CoherenceChecker, CoherenceDecision, CoherenceGate, CoherenceOutcome, GateConfig, GateContext,
  RegimeLabel, ThresholdGate, simple_context}`.

## Related

- `../neural-trader-core` — `MarketEvent`, market data normalization.
- `../neural-trader-coherence` — coherence/regime detection inputs.
- `../ruvector-attention` — attention model used by `attention_scalper`.
