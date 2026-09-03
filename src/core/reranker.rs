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
use std::collections::HashMap;

pub const F_EMIT: usize = 0;
pub const F_LM: usize = 1;
pub const F_LEN: usize = 2;
pub const F_LEX: usize = 3;
/// Corpus frequency of the candidate (log-scaled).  Resolves vowel-length and
/// schwa ambiguity the akshara LM cannot: kal -> कल vs काल is a word-frequency
/// question, not an akshara-sequence question (E3).
pub const F_FREQ: usize = 4;
/// Number of features.
pub const NUM_FEATURES: usize = 5;

#[derive(Debug, Clone)]
pub struct Reranker {
    pub weights: [f64; NUM_FEATURES],
    lexicon: Option<RomanLexicon>,
    freq: Option<HashMap<String, u32>>,
}

impl Default for Reranker {
    fn default() -> Self {
        // Start from the generative balance (emit=1, lm=1) plus zero extras.
        Self {
            weights: [1.0, 1.0, 0.0, 0.0, 0.0],
            lexicon: None,
            freq: None,
        }
    }
}

impl Reranker {
    pub fn new(weights: [f64; NUM_FEATURES], lexicon: Option<RomanLexicon>) -> Self {
        Self { weights, lexicon, freq: None }
    }

    pub fn with_lexicon(mut self, lexicon: Option<RomanLexicon>) -> Self {
        self.lexicon = lexicon;
        self
    }

    pub fn with_freq(mut self, freq: Option<HashMap<String, u32>>) -> Self {
        self.freq = freq;
        self
    }

    /// Feature vector for a candidate (all higher = better).
    pub fn features(&self, roman: &str, cand: &DecodedCandidate) -> [f64; NUM_FEATURES] {
        let in_lex = self
            .lexicon
            .as_ref()
            .map(|lx| lx.has_pair(roman, &cand.dev))
            .unwrap_or(false);
        let freq_feat = self
            .freq
            .as_ref()
            .and_then(|f| f.get(&cand.dev))
            .map(|&c| (1.0 + c as f64).ln() / 100.0f64.ln()) // ~[0,1] up to 1e8
            .unwrap_or(0.0);
        [
            -cand.emit,                     // lower emission cost = better
            -cand.lm,                       // lower LM cost = better
            -(cand.akshara_count as f64),   // prefer fewer aksharas (sign tunable)
            if in_lex { 1.0 } else { 0.0 }, // exact corpus word
            freq_feat.min(1.0),             // corpus frequency of the word
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
    ["emission", "lm", "length", "lexicon", "frequency"]
}
