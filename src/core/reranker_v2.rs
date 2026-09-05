// File: src/core/reranker_v2.rs
//
// Reranker v2: the trained softmax reranker validated offline at 81.69%
// native top-1 (vs 80.98% heuristic) — see docs/plans/2026-09-03-accuracy-
// experiments.md and the generated weights in `reranker_v2_weights.rs`.
//
// Score model, identical to data/pipeline/reranker_v2.py:
//   1. raw feature vector per candidate (order fixed, N_FEATS entries)
//   2. standardized with embedded MEAN/STD (fit on the valid dump)
//   3. score = W · x_std                                  (higher = better)
//   4. blended with the shipped heuristic per case:
//        s = (1-γ)·(−z(heur)) + γ·z(score),  γ = GAMMA
//      where z is the per-case z-score over the candidate list. γ=0
//      reproduces the heuristic ranking exactly; γ=1 is the pure model.
//
// Feature order (indices into the arrays):
//   0 emit  1 lm  2 akshara_count  3 decoder_rank  4 heuristic  5 heuristic_rank
//   6 log1p(freq)  7 freq_rank_pct  8 in_vocab  9 len(dev)  10 matra_total
//   11..20 matra profile (10)  21 nasals  22 visarga  23 halants
//   24 vowel_initial  25 ends_matra  26 ends_nasal_visarga  27 len(roman)
//   28..51 short-word suffix agreement (24, data-derived table)

use crate::core::decoder::DecodedCandidate;
use crate::core::reranker_v2_weights::{
    GAMMA, LM_W, MEAN, N_FEATS, SHORT_WORDS, STD, VOCAB_W, W,
};
use std::collections::HashMap;

const MATRAS: [char; 10] = [
    '\u{093E}', // ा
    '\u{093F}', // ि
    '\u{0940}', // ी
    '\u{0941}', // ु
    '\u{0942}', // ू
    '\u{0947}', // े
    '\u{0948}', // ै
    '\u{094B}', // ो
    '\u{094C}', // ौ
    '\u{0943}', // ृ
];

// Suffix-agreement features come from the data-derived SHORT_WORDS table
// (reranker_v2_weights.rs): frequent short Devanagari words (postpositions
// and clitics dominate) with the roman spellings users actually typed for
// them. No hardcoded linguistics — the table and its slot order are
// generated together with the weights.

/// Word-frequency ranks (position in the vocabulary sorted by frequency,
/// descending) — the model's freq_rank_pct feature. Built once from the
/// vocabulary map.
pub struct FreqRanks {
    pub rank_of: HashMap<String, usize>,
    pub total: usize,
}

/// Everything rerank() needs from the corpus vocabulary.
pub struct RerankerV2Data {
    pub freq: HashMap<String, u32>,
    pub ranks: FreqRanks,
}

impl RerankerV2Data {
    /// Build from the serialized vocabulary (data/word_freq_text.bin).
    /// None if the vocabulary is unavailable (e.g. WASM without fetch).
    pub fn from_bin_bytes(bytes: &[u8]) -> Option<Self> {
        let freq: HashMap<String, u32> = bincode::deserialize(bytes).ok()?;
        let ranks = FreqRanks::from_freq_map(&freq);
        Some(Self { freq, ranks })
    }
}

impl FreqRanks {
    pub fn from_freq_map(freq: &HashMap<String, u32>) -> Self {
        let mut by_freq: Vec<(&String, u32)> =
            freq.iter().map(|(w, &c)| (w, c)).collect();
        by_freq.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        let rank_of: HashMap<String, usize> = by_freq
            .into_iter()
            .enumerate()
            .map(|(i, (w, _))| (w.clone(), i))
            .collect();
        let total = rank_of.len();
        Self { rank_of, total }
    }
}

fn zscore(xs: &mut [f64]) {
    let n = xs.len() as f64;
    if n < 2.0 {
        return;
    }
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let std = var.sqrt() + 1e-9;
    for x in xs.iter_mut() {
        *x = (*x - mean) / std;
    }
}

/// Rank candidates by the v2 blended score. Input order is irrelevant — the
/// decoder-rank feature is recomputed with the canonical key
/// `emit + 1.0·lm` (the dump protocol). Returns (dev, blended score),
/// sorted best-first.
pub fn rerank(
    roman: &str,
    candidates: &[DecodedCandidate],
    freq: &HashMap<String, u32>,
    ranks: &FreqRanks,
) -> Vec<(String, f64)> {
    if candidates.is_empty() {
        return vec![];
    }

    // Canonical decoder order: emit + 1.0·lm ascending (dump protocol).
    let mut order: Vec<&DecodedCandidate> = candidates.iter().collect();
    order.sort_by(|a, b| {
        (a.emit + a.lm)
            .total_cmp(&(b.emit + b.lm))
            .then(a.dev.cmp(&b.dev))
    });

    // Raw heuristic per candidate (lower = better), then its rank.
    let heur: Vec<f64> = order
        .iter()
        .map(|c| {
            let f = freq.get(&c.dev).copied().unwrap_or(0);
            c.emit + LM_W * c.lm - VOCAB_W * (1.0 + f as f64).ln()
        })
        .collect();
    let mut heur_order: Vec<usize> = (0..order.len()).collect();
    heur_order.sort_by(|a, b| {
        heur[*a]
            .total_cmp(&heur[*b])
            .then(order[*a].dev.cmp(&order[*b].dev))
    });
    let mut heur_rank = vec![0.0f64; order.len()];
    for (r, i) in heur_order.into_iter().enumerate() {
        heur_rank[i] = r as f64;
    }

    // Raw feature vectors (standardized below).
    let n = order.len();
    let mut xs: Vec<[f64; N_FEATS]> = Vec::with_capacity(n);
    for (i, c) in order.iter().enumerate() {
        let f = freq.get(&c.dev).copied().unwrap_or(0);
        let rank_pct = ranks.rank_of.get(&c.dev).copied().unwrap_or(ranks.total) as f64
            / ranks.total.max(1) as f64;
        let mut feats = [0.0f64; N_FEATS];
        feats[0] = c.emit;
        feats[1] = c.lm;
        feats[2] = c.akshara_count as f64;
        feats[3] = i as f64;
        feats[4] = heur[i];
        feats[5] = heur_rank[i];
        feats[6] = (1.0 + f as f64).ln();
        feats[7] = rank_pct;
        feats[8] = if f > 0 { 1.0 } else { 0.0 };
        feats[9] = c.dev.chars().count() as f64;
        let matra_total: f64 = MATRAS.iter().map(|m| c.dev.matches(*m).count() as f64).sum();
        feats[10] = matra_total;
        for (k, m) in MATRAS.iter().enumerate() {
            feats[11 + k] = c.dev.matches(*m).count() as f64;
        }
        feats[21] = (c.dev.matches('\u{0902}').count() + c.dev.matches('\u{0901}').count()) as f64;
        feats[22] = c.dev.matches('\u{0903}').count() as f64;
        feats[23] = c.dev.matches('\u{094D}').count() as f64;
        let first = c.dev.chars().next();
        feats[24] = first.is_some_and(|ch| "अआइईउऊएऐओऔऋ".contains(ch)) as i32 as f64;
        feats[25] = c
            .dev
            .chars()
            .last()
            .is_some_and(|ch| MATRAS.contains(&ch)) as i32 as f64;
        feats[26] = c.dev.chars().last().is_some_and(|ch| ch == '\u{0902}' || ch == '\u{0901}' || ch == '\u{0903}') as i32 as f64;
        feats[27] = roman.chars().count() as f64;
        for (k, (word, variants)) in SHORT_WORDS.iter().enumerate() {
            feats[28 + k] = (c.dev.ends_with(word)
                && variants.iter().any(|r| !r.is_empty() && roman.ends_with(r)))
                as i32 as f64;
        }
        xs.push(feats);
    }

    // Standardize + score + blend.
    let mut scores: Vec<f64> = Vec::with_capacity(n);
    let mut heur_std: Vec<f64> = Vec::with_capacity(n);
    for (i, x) in xs.iter().enumerate() {
        let mut s = 0.0f64;
        for k in 0..N_FEATS {
            s += W[k] * (x[k] - MEAN[k]) / STD[k];
        }
        scores.push(s);
        heur_std.push((x[4] - MEAN[4]) / STD[4]);
    }
    zscore(&mut heur_std);
    zscore(&mut scores);
    let mut blended: Vec<(String, f64)> = order
        .iter()
        .zip(scores.into_iter())
        .zip(heur_std.into_iter())
        .map(|((c, s), zh)| (c.dev.clone(), (1.0 - GAMMA) * (-zh) + GAMMA * s))
        .collect();
    blended.sort_by(|a, b| b.1.total_cmp(&a.1));
    blended
}
