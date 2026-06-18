//! Trainable multiresolution feature tables and their gradient accumulator
//! (ADR-258 §6.1, §6.4).
//!
//! Tables are stored in memory as one contiguous `Vec<f32>` per level (row-major
//! `[rows, F]`), initialized with small uniform noise as in Instant-NGP. The
//! same shape backs the [`GradAccum`], so `apply` is a single fused AXPY per
//! level — the persistence-friendly update used by the GNN self-learning loop.

use crate::config::HashEncConfig;
use crate::rng::SplitMix64;
use std::io::{self, Read, Write};
use std::path::Path;

/// Trainable feature tables — one per resolution level.
#[derive(Clone, Debug)]
pub struct FeatureTables {
    pub(crate) cfg: HashEncConfig,
    /// `levels[l]` has length `level_rows(l) * F`.
    levels: Vec<Vec<f32>>,
    /// Cached resolutions and dense flags per level.
    res: Vec<u32>,
    dense: Vec<bool>,
}

const TABLE_MAGIC: u32 = 0x5248_4531; // "RHE1"

impl FeatureTables {
    /// Allocate and randomly initialize tables for `cfg`.
    pub fn new(cfg: &HashEncConfig) -> Self {
        let f = cfg.features_per_level;
        let mut rng = SplitMix64::new(cfg.seed);
        let mut levels = Vec::with_capacity(cfg.levels);
        let mut res = Vec::with_capacity(cfg.levels);
        let mut dense = Vec::with_capacity(cfg.levels);
        for l in 0..cfg.levels {
            let rows = cfg.level_rows(l);
            let mut t = vec![0.0f32; rows * f];
            for v in &mut t {
                *v = rng.next_signed(1e-4); // Instant-NGP init range.
            }
            levels.push(t);
            res.push(cfg.resolution(l));
            dense.push(cfg.level_is_dense(l));
        }
        Self {
            cfg: cfg.clone(),
            levels,
            res,
            dense,
        }
    }

    #[inline]
    pub fn features_per_level(&self) -> usize {
        self.cfg.features_per_level
    }

    #[inline]
    pub fn resolution(&self, level: usize) -> u32 {
        self.res[level]
    }

    #[inline]
    pub fn is_dense(&self, level: usize) -> bool {
        self.dense[level]
    }

    /// Read-only feature slice (`F` values) for a `(level, row)`.
    #[inline]
    pub fn row(&self, level: usize, row: usize) -> &[f32] {
        let f = self.cfg.features_per_level;
        let base = row * f;
        &self.levels[level][base..base + f]
    }

    /// Mutable feature slice for a `(level, row)`.
    #[inline]
    pub fn row_mut(&mut self, level: usize, row: usize) -> &mut [f32] {
        let f = self.cfg.features_per_level;
        let base = row * f;
        &mut self.levels[level][base..base + f]
    }

    /// Total trainable parameter count (sum over levels of `rows * F`).
    pub fn param_count(&self) -> usize {
        self.levels.iter().map(|t| t.len()).sum()
    }

    /// In-memory byte footprint of the tables.
    pub fn byte_size(&self) -> usize {
        self.param_count() * std::mem::size_of::<f32>()
    }

    /// Serialize tables to a file (dependency-free persistence; a live mmap
    /// backend is the Phase 3 upgrade described in ADR-258 §6.4).
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut w = io::BufWriter::new(std::fs::File::create(path)?);
        w.write_all(&TABLE_MAGIC.to_le_bytes())?;
        w.write_all(&(self.cfg.levels as u32).to_le_bytes())?;
        w.write_all(&(self.cfg.features_per_level as u32).to_le_bytes())?;
        for t in &self.levels {
            w.write_all(&(t.len() as u64).to_le_bytes())?;
            let bytes: &[u8] = bytemuck_cast(t);
            w.write_all(bytes)?;
        }
        w.flush()
    }

    /// Load tables previously written by [`save`] into `cfg`-shaped buffers.
    pub fn load(cfg: &HashEncConfig, path: &Path) -> io::Result<Self> {
        let mut r = io::BufReader::new(std::fs::File::open(path)?);
        let mut u32buf = [0u8; 4];
        r.read_exact(&mut u32buf)?;
        if u32::from_le_bytes(u32buf) != TABLE_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        let mut me = Self::new(cfg);
        r.read_exact(&mut u32buf)?; // levels
        r.read_exact(&mut u32buf)?; // F
        for t in &mut me.levels {
            let mut u64buf = [0u8; 8];
            r.read_exact(&mut u64buf)?;
            let len = u64::from_le_bytes(u64buf) as usize;
            if len != t.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "shape mismatch"));
            }
            let bytes: &mut [u8] = bytemuck_cast_mut(t);
            r.read_exact(bytes)?;
        }
        Ok(me)
    }
}

/// Gradient accumulator mirroring [`FeatureTables`] shape.
#[derive(Clone, Debug)]
pub struct GradAccum {
    f: usize,
    levels: Vec<Vec<f32>>,
}

impl GradAccum {
    pub fn new(tables: &FeatureTables) -> Self {
        Self {
            f: tables.cfg.features_per_level,
            levels: tables.levels.iter().map(|t| vec![0.0f32; t.len()]).collect(),
        }
    }

    /// Accumulate `val` into feature `feat` of `(level, row)`.
    #[inline]
    pub fn add(&mut self, level: usize, row: usize, feat: usize, val: f32) {
        self.levels[level][row * self.f + feat] += val;
    }

    /// Fused SGD update: `tables -= lr * grad`, then zero the accumulator.
    pub fn apply(&mut self, tables: &mut FeatureTables, lr: f32) {
        for (l, g) in self.levels.iter_mut().enumerate() {
            let t = &mut tables.levels[l];
            for i in 0..t.len() {
                t[i] -= lr * g[i];
                g[i] = 0.0;
            }
        }
    }

    pub fn zero(&mut self) {
        for g in &mut self.levels {
            for v in g {
                *v = 0.0;
            }
        }
    }

    /// Accumulated gradient value at `(level, row, feat)` (used by tests).
    #[inline]
    pub fn value(&self, level: usize, row: usize, feat: usize) -> f32 {
        self.levels[level][row * self.f + feat]
    }

    /// L2 norm of the accumulated gradient (diagnostics).
    pub fn l2_norm(&self) -> f32 {
        self.levels
            .iter()
            .flat_map(|g| g.iter())
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
    }
}

// --- minimal, safe f32<->u8 slice casting (avoids a bytemuck dependency) ---

fn bytemuck_cast(s: &[f32]) -> &[u8] {
    // Safety: f32 has no padding/invalid bit patterns; length scaled by 4.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

fn bytemuck_cast_mut(s: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, std::mem::size_of_val(s)) }
}
