# neural-trader-strategies/src

Source modules for the venue-agnostic strategy + risk-gate runtime.

## Files

- `lib.rs` — declares `Strategy` trait, re-exports all public types.
- `intent.rs` — `Intent`, `Action`, `Side` — canonical, venue-agnostic order representation.
- `risk.rs` — `RiskGate` and supporting `RiskConfig` / `RiskDecision` / `RejectReason` / `PortfolioState` / `Position`. Wraps every
  `Intent` and enforces position cap, daily-loss kill, concentration, min-edge, and the live-trade env flag.
- `ev_kelly.rs` — `ExpectedValueKelly` strategy + config; first concrete implementer.
- `attention_scalper.rs` — `AttentionScalper` using attention-model signals from `ruvector-attention`.
- `coherence_arb.rs` — `CoherenceArb` exploiting coherence/regime mispricings.
- `coherence_bridge.rs` — `CoherenceChecker`, `CoherenceGate`, `ThresholdGate`, and surrounding regime / decision types that
  bridge `neural-trader-coherence` outputs into strategy gates.
