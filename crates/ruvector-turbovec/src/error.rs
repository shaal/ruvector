//! Error type for the TurboVec index.

use thiserror::Error;

/// Errors produced by [`crate::TurboVecIndex`] / [`crate::IdMapIndex`].
#[derive(Debug, Error)]
pub enum TurboVecError {
    /// A vector's length did not match the index dimension.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },

    /// `dim` was zero at construction.
    #[error("dimension must be > 0")]
    ZeroDim,

    /// An external id was reused in `add_with_ids`.
    #[error("duplicate external id: {0}")]
    DuplicateId(u64),
}

/// Crate result alias.
pub type Result<T> = std::result::Result<T, TurboVecError>;
