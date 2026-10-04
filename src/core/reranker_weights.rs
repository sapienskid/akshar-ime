// File: src/core/reranker_weights.rs
//
// Structural constants for the reranker, and blend defaults for the no-model path.
//
// Trained parameters (dense weights, feature mean/std, sparse table and its
// scale) live only in data/akshar.model. They used to be duplicated here as
// compiled-in constants; they are gone because they went stale silently.

pub const DENSE_DIM: usize = 29;
pub const HASH_SIZE: usize = 1048576;

/// Dequantisation scale for the container's i8 sparse table. The trained value
/// is `UnifiedModel::sparse_scale`; 1.0 is the unit-scale fallback for the
/// no-model path.
pub const DEFAULT_SPARSE_SCALE: f64 = 1.0;

// No-model defaults: used only when no container is loaded. `make model`
// writes the calibrated blend into UnifiedModel::gamma_auto.
pub const LM_W: f64 = 0.85;
pub const VOCAB_W: f64 = 0.75;
pub const GAMMA: f64 = 0.3;
pub const CTX_W: f64 = 0.5;
