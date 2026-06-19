//! Tiny dependency-free, deterministic PRNG (splitmix64) for reproducible
//! table and projection initialization. Keeps the crate WASM-friendly with no
//! external RNG dependency.

/// Deterministic splitmix64 generator.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // 24 mantissa bits.
        ((self.next_u64() >> 40) as f32) * (1.0 / (1u32 << 24) as f32)
    }

    /// Uniform f32 in [-mag, mag).
    #[inline]
    pub fn next_signed(&mut self, mag: f32) -> f32 {
        (self.next_f32() * 2.0 - 1.0) * mag
    }

    /// Approx. standard normal via Box-Muller.
    #[inline]
    pub fn next_normal(&mut self) -> f32 {
        let u1 = (self.next_f32()).max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}
