// File: src/core/reranker.rs
//
// Discriminative reranking over the generative decoder's k-best list.
//
// The generative beam produces k candidates with decomposed scores (emission
// and LM).  A small log-linear reranker scores each candidate as a weighted sum
// of interpretable features, with weights tuned by MERT (coordinate ascent
// minimising top-1 error) on a held-out dev split.  This is the classical
// "generative base + discriminative rerank" recipe that dominated the NEWS
// transliteration shared tasks.

use crate::core::decoder::DecodedCandidate;
use crate::core::lexicon::RomanLexicon;

pub const F_EMIT: usize = 0;
pub const F_LM: usize = 1;
pub const F_LEN: usize = 2;
pub const F_LEX: usize = 3;
/// Number of features.
pub const NUM_FEATURES: usize = 4;

#[derive(Debug, Clone)]
pub struct Reranker {
    pub weights: [f64; NUM_FEATURES],
    lexicon: Option<RomanLexicon>,
}

impl Default for Reranker {
    fn default() -> Self {
        // Start from the generative balance (emit=1, lm=1) plus zero extras.
        Self {
            weights: [1.0, 1.0, 0.0, 0.0],
            lexicon: None,
        }
    }
}

impl Reranker {
    pub fn new(weights: [f64; NUM_FEATURES], lexicon: Option<RomanLexicon>) -> Self {
        Self { weights, lexicon }
    }

    pub fn with_lexicon(mut self, lexicon: Option<RomanLexicon>) -> Self {
        self.lexicon = lexicon;
        self
    }

    /// Feature vector for a candidate (all higher = better).
    pub fn features(&self, roman: &str, cand: &DecodedCandidate) -> [f64; NUM_FEATURES] {
        let in_lex = self
            .lexicon
            .as_ref()
            .map(|lx| lx.has_pair(roman, &cand.dev))
            .unwrap_or(false);
        [
            -cand.emit,                       // lower emission cost = better
            -cand.lm,                         // lower LM cost = better
            -(cand.akshara_count as f64),     // prefer fewer aksharas (sign tunable)
            if in_lex { 1.0 } else { 0.0 },  // exact corpus word
        ]
    }

    /// Rerank the decoder's k-best candidates, returning (devanagari, score).
    pub fn rerank(&self, roman: &str, candidates: Vec<DecodedCandidate>) -> Vec<(String, f64)> {
        let mut scored: Vec<(String, f64)> = candidates
            .into_iter()
            .map(|c| {
                let f = self.features(roman, &c);
                let score: f64 = self.weights.iter().zip(f.iter()).map(|(w, x)| w * x).sum();
                (c.dev, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored
    }
}

/// Feature names for diagnostics / serialisation of a trained model.
pub fn feature_names() -> [&'static str; NUM_FEATURES] {
    ["emission", "lm", "length", "lexicon"]
}
