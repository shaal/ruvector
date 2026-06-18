//! Spatial hashing and grid indexing (ADR-258 §6.1).
//!
//! Implements the Instant-NGP spatial hash `h(x) = (⊕_i x_i·π_i) mod T` with
//! the canonical large primes, plus collision-free dense indexing for coarse
//! levels whose grid fits within `T`.

/// Primes for dimensions up to 7 (π_1 = 1 by convention).
pub const PRIMES: [u32; 7] = [
    1,
    2_654_435_761,
    805_459_861,
    3_674_653_429,
    2_097_192_037,
    1_434_869_437,
    2_165_219_737,
];

/// Spatial hash of integer grid corner coordinates into `[0, 2^log2_t)`.
#[inline]
pub fn spatial_hash(corner: &[u32], log2_t: u8) -> usize {
    let mut h: u32 = 0;
    for (i, &c) in corner.iter().enumerate() {
        h ^= c.wrapping_mul(PRIMES[i]);
    }
    (h as usize) & ((1usize << log2_t) - 1)
}

/// Dense (collision-free) row index for a coarse level: mixed-radix encoding of
/// corner coordinates with stride `(N+1)`.
#[inline]
pub fn dense_index(corner: &[u32], grid_dim: u32) -> usize {
    let stride = grid_dim as usize + 1;
    let mut idx = 0usize;
    let mut mul = 1usize;
    for &c in corner {
        idx += (c as usize) * mul;
        mul *= stride;
    }
    idx
}

/// Resolve a corner to a table row, choosing dense vs hashed indexing.
#[inline]
pub fn row_index(corner: &[u32], grid_dim: u32, log2_t: u8, dense: bool) -> usize {
    if dense {
        dense_index(corner, grid_dim)
    } else {
        spatial_hash(corner, log2_t)
    }
}
