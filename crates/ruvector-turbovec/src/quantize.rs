//! Lloyd–Max scalar quantization for the canonical unit-Gaussian marginal,
//! plus tight bit-packing.
//!
//! After the randomized Hadamard rotation (reused from `ruvector-rabitq`) and
//! TQ+ per-coordinate calibration (`crate::calibrate`), each coordinate is
//! approximately `N(0, 1)`. We therefore quantize against **codebook-free**,
//! data-independent Lloyd–Max centroids for the standard normal — no k-means,
//! no training pass, online ingest (ADR-194 §T2).
//!
//! Quantization is nearest-centroid, which is exactly Lloyd–Max decision
//! regions (boundaries are the midpoints between adjacent centroids). Storing
//! only the centroid table keeps the encoder a few `f32` comparisons per
//! coordinate.

/// Supported quantization widths. `One` is included as a correctness/recall
/// baseline against `ruvector-rabitq`'s 1-bit path; `Two`/`Four` are the
/// production targets of ADR-194.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BitWidth {
    /// 1 bit / coord (2 levels).
    One,
    /// 2 bits / coord (4 levels).
    Two,
    /// 4 bits / coord (16 levels).
    Four,
}

impl BitWidth {
    /// Bits stored per coordinate.
    #[inline]
    pub const fn bits(self) -> usize {
        match self {
            BitWidth::One => 1,
            BitWidth::Two => 2,
            BitWidth::Four => 4,
        }
    }

    /// Number of reconstruction levels (`2^bits`).
    #[inline]
    pub const fn levels(self) -> usize {
        1usize << self.bits()
    }

    /// Lloyd–Max centroids for the standard normal `N(0, 1)`, ascending.
    /// (Max, *Quantizing for Minimum Distortion*, IRE 1960.)
    #[inline]
    pub fn centroids(self) -> &'static [f32] {
        match self {
            // ±sqrt(2/π)
            BitWidth::One => &[-0.797_884_6, 0.797_884_6],
            BitWidth::Two => &[-1.510_4, -0.452_8, 0.452_8, 1.510_4],
            BitWidth::Four => &[
                -2.732_6, -2.069_0, -1.618_0, -1.256_2, -0.942_4, -0.656_8, -0.388_1, -0.128_4,
                0.128_4, 0.388_1, 0.656_8, 0.942_4, 1.256_2, 1.618_0, 2.069_0, 2.732_6,
            ],
        }
    }

    /// Packed code bytes for `dim` coordinates: `ceil(dim * bits / 8)`.
    #[inline]
    pub const fn code_bytes(self, dim: usize) -> usize {
        (dim * self.bits()).div_ceil(8)
    }
}

/// Quantize one calibrated coordinate to its nearest centroid index.
///
/// Linear scan over `levels` (≤ 16) — branch-predictable and trivially
/// vectorizable; the FastScan SIMD kernel (ADR-194 §T5, future milestone)
/// will replace this with a nibble LUT, but the *result* must stay identical.
#[inline]
pub fn quantize_coord(centroids: &[f32], z: f32) -> u8 {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, &c) in centroids.iter().enumerate() {
        let d = (z - c).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best as u8
}

/// Pack per-coordinate code indices (`0..levels`) tightly, MSB-first within
/// each byte, into `code_bytes(dim)` bytes.
pub fn pack(codes: &[u8], bw: BitWidth) -> Vec<u8> {
    let bits = bw.bits();
    let mut out = vec![0u8; bw.code_bytes(codes.len())];
    let mut bit_pos = 0usize;
    for &code in codes {
        for b in (0..bits).rev() {
            let bit = (code >> b) & 1;
            if bit != 0 {
                let byte = bit_pos / 8;
                let off = 7 - (bit_pos % 8);
                out[byte] |= 1 << off;
            }
            bit_pos += 1;
        }
    }
    out
}

/// Inverse of [`pack`]: recover `dim` code indices from packed bytes.
pub fn unpack(packed: &[u8], dim: usize, bw: BitWidth) -> Vec<u8> {
    let bits = bw.bits();
    let mut out = vec![0u8; dim];
    let mut bit_pos = 0usize;
    for slot in out.iter_mut() {
        let mut code = 0u8;
        for _ in 0..bits {
            let byte = bit_pos / 8;
            let off = 7 - (bit_pos % 8);
            let bit = (packed[byte] >> off) & 1;
            code = (code << 1) | bit;
            bit_pos += 1;
        }
        *slot = code;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centroids_are_sorted_and_sized() {
        for bw in [BitWidth::One, BitWidth::Two, BitWidth::Four] {
            let c = bw.centroids();
            assert_eq!(c.len(), bw.levels());
            assert!(c.windows(2).all(|w| w[0] < w[1]), "{bw:?} not ascending");
        }
    }

    #[test]
    fn pack_unpack_roundtrip_all_widths() {
        for bw in [BitWidth::One, BitWidth::Two, BitWidth::Four] {
            let dim = 37; // deliberately not byte-aligned
            let codes: Vec<u8> = (0..dim).map(|i| (i % bw.levels()) as u8).collect();
            let packed = pack(&codes, bw);
            assert_eq!(packed.len(), bw.code_bytes(dim));
            let back = unpack(&packed, dim, bw);
            assert_eq!(codes, back, "roundtrip failed for {bw:?}");
        }
    }

    #[test]
    fn quantize_picks_nearest_centroid() {
        let c = BitWidth::Four.centroids();
        // A value sitting exactly on a centroid must map to that centroid.
        for (i, &cv) in c.iter().enumerate() {
            assert_eq!(quantize_coord(c, cv) as usize, i);
        }
        // Far-positive saturates to the top bucket, far-negative to the bottom.
        assert_eq!(quantize_coord(c, 99.0) as usize, c.len() - 1);
        assert_eq!(quantize_coord(c, -99.0) as usize, 0);
    }
}
