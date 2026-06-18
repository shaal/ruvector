//! Per-block quantization tiers — the ADR-004 bridge.
//!
//! ADR-004 defines a 3-tier adaptive scheme (FP16 hot → 4-bit warm → 2-bit
//! archive). In the paged design the *block* is the unit of both residency and
//! precision, so the tier lives on each [`super::PhysicalBlock`] and a
//! [`BlockQuantizer`] codec compresses/decompresses block payloads.
//!
//! The pool stores block payloads as `f32` for kernel simplicity; a quantizer
//! is a logical codec that decides how many *bits of information* a sealed
//! block retains. Demoting a block re-quantizes its f32 payload in place
//! (lossy), so a block's tier records how aggressively it has been compressed.
//! This keeps the allocator GPU-free while letting ADR-004's policy drive
//! precision one block at a time.

use serde::{Deserialize, Serialize};

/// Precision tier of a physical block, mirroring ADR-004's three tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantTier {
    /// Hot buffer: full precision, no quantization overhead. ADR-004 Tier 1.
    Fp16,
    /// Warm zone: 4-bit KIVI-style. ADR-004 Tier 2.
    Int4,
    /// Deep archive: 2-bit KIVI/SQuat. ADR-004 Tier 3.
    Int2,
}

impl QuantTier {
    /// Bits retained per stored element. Used for memory accounting and to
    /// estimate the compression ratio vs. an FP16 baseline.
    #[inline]
    pub fn bits(self) -> u8 {
        match self {
            QuantTier::Fp16 => 16,
            QuantTier::Int4 => 4,
            QuantTier::Int2 => 2,
        }
    }

    /// Compression ratio of this tier relative to FP16 (ignoring scale
    /// overhead). FP16 → 1.0, Int4 → 4.0, Int2 → 8.0.
    #[inline]
    pub fn compression_ratio(self) -> f32 {
        16.0 / self.bits() as f32
    }

    /// The next more-aggressive tier, or `None` if already at the floor. ADR-004
    /// staleness/quality policy calls this to demote a cooling block.
    #[inline]
    pub fn demote(self) -> Option<QuantTier> {
        match self {
            QuantTier::Fp16 => Some(QuantTier::Int4),
            QuantTier::Int4 => Some(QuantTier::Int2),
            QuantTier::Int2 => None,
        }
    }
}

impl Default for QuantTier {
    fn default() -> Self {
        QuantTier::Fp16
    }
}

/// Codec applied to a block payload when it transitions between tiers.
///
/// Implementors bridge the existing ADR-004 quantizers (KIVI/SQuat/KVQuant). The
/// codec operates on the flat `f32` payload of a single block, which is exactly
/// `block_size * num_kv_heads * head_dim` elements.
pub trait BlockQuantizer: Send + Sync {
    /// Re-quantize `payload` in place to `target` precision. Higher-precision →
    /// lower-precision is lossy; the reverse is a no-op (information is already
    /// gone) and must not fabricate precision.
    fn requantize(&self, payload: &mut [f32], from: QuantTier, target: QuantTier);

    /// Human-readable codec name for telemetry.
    fn name(&self) -> &'static str;
}

/// No-op quantizer used by default and in tests: keeps the f32 payload bit-exact
/// regardless of the nominal tier. Useful for validating *paging* behavior in
/// isolation from *precision* behavior (a deliberate ADR-258 separation of
/// concerns — paging correctness must not depend on a real codec).
#[derive(Debug, Default, Clone, Copy)]
pub struct IdentityQuantizer;

impl BlockQuantizer for IdentityQuantizer {
    #[inline]
    fn requantize(&self, _payload: &mut [f32], _from: QuantTier, _target: QuantTier) {
        // Intentionally lossless: tier is tracked for accounting only.
    }

    #[inline]
    fn name(&self) -> &'static str {
        "identity"
    }
}

/// A simple uniform-rounding quantizer that *simulates* tier precision loss by
/// snapping each element to the nearest representable level for the target tier.
///
/// This is a faithful-enough stand-in for KIVI's per-channel/per-token schemes
/// at the block level for benchmarking the precision/quality trade-off without
/// pulling the full ADR-004 quantizer machinery into the allocator. Production
/// builds swap in the real KIVI/SQuat codecs behind the same trait.
#[derive(Debug, Default, Clone, Copy)]
pub struct UniformQuantizer;

impl BlockQuantizer for UniformQuantizer {
    fn requantize(&self, payload: &mut [f32], from: QuantTier, target: QuantTier) {
        if target == from || payload.is_empty() {
            return;
        }
        // Only quantize when reducing precision; never invent bits.
        if target.bits() >= from.bits() {
            return;
        }
        // Per-block symmetric uniform quantization: one scale for the whole
        // block keeps the codec O(n) and branch-light. KIVI's asymmetric
        // per-channel keys / per-token values is what the production codec adds.
        let levels = ((1u32 << target.bits()) - 1) as f32;
        let max_abs = payload.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        if max_abs == 0.0 {
            return;
        }
        let scale = max_abs / levels;
        let inv = 1.0 / scale;
        for x in payload.iter_mut() {
            let q = (*x * inv).round().clamp(-levels, levels);
            *x = q * scale;
        }
    }

    fn name(&self) -> &'static str {
        "uniform"
    }
}

#[cfg(test)]
mod quant_tests {
    use super::*;

    #[test]
    fn tier_demote_chain() {
        assert_eq!(QuantTier::Fp16.demote(), Some(QuantTier::Int4));
        assert_eq!(QuantTier::Int4.demote(), Some(QuantTier::Int2));
        assert_eq!(QuantTier::Int2.demote(), None);
    }

    #[test]
    fn compression_ratios() {
        assert_eq!(QuantTier::Fp16.compression_ratio(), 1.0);
        assert_eq!(QuantTier::Int4.compression_ratio(), 4.0);
        assert_eq!(QuantTier::Int2.compression_ratio(), 8.0);
    }

    #[test]
    fn identity_is_lossless() {
        let mut p = vec![1.0, -2.5, 3.14, 0.0];
        let orig = p.clone();
        IdentityQuantizer.requantize(&mut p, QuantTier::Fp16, QuantTier::Int2);
        assert_eq!(p, orig);
    }

    #[test]
    fn uniform_never_invents_precision() {
        // Upcasting must be a no-op.
        let mut p = vec![0.1, 0.2, 0.3];
        let before = p.clone();
        UniformQuantizer.requantize(&mut p, QuantTier::Int2, QuantTier::Fp16);
        assert_eq!(p, before);
    }

    #[test]
    fn uniform_reduces_distinct_levels() {
        let mut p: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        UniformQuantizer.requantize(&mut p, QuantTier::Fp16, QuantTier::Int2);
        // 2-bit symmetric → at most ~7 distinct magnitudes; assert heavy
        // dedup vs the 256 originals.
        let mut distinct: Vec<i64> = p.iter().map(|x| (x * 1e6) as i64).collect();
        distinct.sort_unstable();
        distinct.dedup();
        assert!(distinct.len() <= 16, "got {} levels", distinct.len());
    }
}
