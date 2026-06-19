//! Self-learning utilities for the contrastive loop (ADR-258 §6.3).
//!
//! - [`NegativeSampler`] — random, HNSW-hard (mid-rank "near but wrong"), or a
//!   mix; hard negatives sharpen the decision boundary and speed convergence.
//! - [`TemperatureSchedule`] — cosine annealing of the InfoNCE temperature from
//!   a soft start to a sharp finish as training matures.

use crate::rng::SplitMix64;

/// How negatives are drawn for a contrastive step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NegativeSampler {
    /// Uniformly random over the corpus.
    Random,
    /// Mid-rank HNSW candidates in `[band.0, band.1)` — semantically near but
    /// not relevant; the most informative negatives.
    HnswHard { band: (usize, usize) },
    /// `frac` fraction hard (mid-rank), the rest random.
    Mixed { band: (usize, usize), hard_frac: f32 },
}

impl NegativeSampler {
    /// Sample `n` negative indices given a ranked candidate list (best-first,
    /// e.g. from an HNSW `ef`-search) and corpus size `n_items`. `exclude` (the
    /// positives) are skipped. Deterministic given `rng`.
    pub fn sample(
        &self,
        ranked: &[usize],
        n_items: usize,
        n: usize,
        exclude: &[usize],
        rng: &mut SplitMix64,
    ) -> Vec<usize> {
        let is_excluded = |x: usize| exclude.contains(&x);
        let mut out = Vec::with_capacity(n);
        match *self {
            NegativeSampler::Random => {
                while out.len() < n {
                    let c = (rng.next_u64() % n_items as u64) as usize;
                    if !is_excluded(c) {
                        out.push(c);
                    }
                }
            }
            NegativeSampler::HnswHard { band } => {
                let (lo, hi) = (band.0.min(ranked.len()), band.1.min(ranked.len()));
                if hi > lo {
                    let mut k = lo;
                    while out.len() < n {
                        let c = ranked[lo + (k - lo) % (hi - lo)];
                        if !is_excluded(c) {
                            out.push(c);
                        }
                        k += 1;
                        if k - lo > 4 * (hi - lo) {
                            break; // avoid infinite loop if band is tiny/excluded
                        }
                    }
                }
                // top up with random if the band couldn't supply enough
                while out.len() < n {
                    let c = (rng.next_u64() % n_items as u64) as usize;
                    if !is_excluded(c) {
                        out.push(c);
                    }
                }
            }
            NegativeSampler::Mixed { band, hard_frac } => {
                let n_hard = ((n as f32) * hard_frac).round() as usize;
                let hard =
                    NegativeSampler::HnswHard { band }.sample(ranked, n_items, n_hard, exclude, rng);
                out.extend(hard);
                let mut excl2 = exclude.to_vec();
                excl2.extend_from_slice(&out);
                let rest =
                    NegativeSampler::Random.sample(ranked, n_items, n - out.len(), &excl2, rng);
                out.extend(rest);
            }
        }
        out
    }
}

/// Cosine-annealed temperature schedule for InfoNCE.
#[derive(Clone, Copy, Debug)]
pub struct TemperatureSchedule {
    pub start: f32,
    pub end: f32,
    pub total_steps: usize,
}

impl TemperatureSchedule {
    pub fn new(start: f32, end: f32, total_steps: usize) -> Self {
        Self {
            start,
            end,
            total_steps: total_steps.max(1),
        }
    }

    /// Temperature at training step `step` (cosine from `start` to `end`).
    pub fn at(&self, step: usize) -> f32 {
        let t = (step.min(self.total_steps) as f32) / (self.total_steps as f32);
        let cos = 0.5 * (1.0 + (std::f32::consts::PI * t).cos()); // 1 -> 0
        self.end + (self.start - self.end) * cos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hard_negatives_come_from_band() {
        let ranked: Vec<usize> = (0..100).collect();
        let mut rng = SplitMix64::new(1);
        let s = NegativeSampler::HnswHard { band: (10, 20) };
        let negs = s.sample(&ranked, 100, 8, &[0, 1, 2, 3], &mut rng);
        assert_eq!(negs.len(), 8);
        assert!(negs.iter().all(|&x| (10..20).contains(&x)));
    }

    #[test]
    fn random_excludes_positives() {
        let ranked: Vec<usize> = (0..50).collect();
        let mut rng = SplitMix64::new(7);
        let negs = NegativeSampler::Random.sample(&ranked, 50, 16, &[5, 6, 7], &mut rng);
        assert_eq!(negs.len(), 16);
        assert!(negs.iter().all(|&x| ![5, 6, 7].contains(&x)));
    }

    #[test]
    fn mixed_has_correct_count() {
        let ranked: Vec<usize> = (0..200).collect();
        let mut rng = SplitMix64::new(3);
        let s = NegativeSampler::Mixed {
            band: (16, 64),
            hard_frac: 0.5,
        };
        let negs = s.sample(&ranked, 200, 16, &[], &mut rng);
        assert_eq!(negs.len(), 16);
    }

    #[test]
    fn temperature_anneals_monotonically() {
        let sched = TemperatureSchedule::new(0.2, 0.05, 100);
        assert!((sched.at(0) - 0.2).abs() < 1e-5);
        assert!((sched.at(100) - 0.05).abs() < 1e-4);
        // strictly decreasing
        let mut prev = sched.at(0);
        for step in (10..=100).step_by(10) {
            let cur = sched.at(step);
            assert!(cur <= prev + 1e-6, "temperature should not increase");
            prev = cur;
        }
    }
}
