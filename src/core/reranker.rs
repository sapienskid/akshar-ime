// File: src/core/reranker.rs
//
// Production Discriminative Reranker: 29 dense shape/frequency/morphology
// features and a 2^20-slot sparse lexicalized hash table.
//
// Accuracy: docs/MANUAL.md §2.3 and docs/experiments/.
//
// Features:
//   0 emit  1 lm  2 akshara_count  3 decoder_rank  4 heuristic  5 heuristic_rank
//   6 log1p(freq)  7 freq_rank_pct  8 in_vocab  9 len(dev)  10 matra_total
//   11..20 matra profile (10)  21 nasals  22 visarga  23 halants
//   24 vowel_initial  25 ends_matra  26 ends_nasal_visarga  27 len(roman)
//   28 morph_effective_log_freq
//
// Sparse features (hashed into 2^20 table):
//   1. Length delta bucket
//   2. (final_akshara x final_roman)
//   3. (first_akshara x first_roman)
//   4. (matra x preceding_consonant)
//   5. (morph_suffix x roman_tail)
//   6. (final_matra x final_roman)
//   7. (penultimate_consonant x final_matra)

use crate::core::akshara;
use crate::core::decoder::DecodedCandidate;
pub use crate::core::reranker_weights::{DENSE_DIM, HASH_SIZE};
use crate::core::reranker_weights::{GAMMA, LM_W, VOCAB_W};
use std::collections::HashMap;

/// How many candidates get full dense+sparse feature extraction.
///
/// The remainder keep their heuristic ordering below the scored block.  50 (the
/// decode depth) disables the cascade.
const DEFAULT_RERANK_FULL_DEPTH: usize = 24;

#[cfg(target_arch = "wasm32")]
fn rerank_full_depth() -> usize {
    DEFAULT_RERANK_FULL_DEPTH
}

#[cfg(not(target_arch = "wasm32"))]
fn rerank_full_depth() -> usize {
    std::env::var("AKSHAR_RERANK_DEPTH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0 && v <= 512)
        .unwrap_or(DEFAULT_RERANK_FULL_DEPTH)
}

pub const MATRAS: [char; 10] = [
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

/// Morphological suffix table for Nepali (34 postpositions, clitics, and verb endings).
///
/// Longest-first order by character count ensures greedy stripping (e.g. गरेको yields को,
/// आएको yields एको).
pub const MORPH_SUFFIXES: [&str; 34] = [
    // 4 chars
    "देखि",
    "थियो",
    "थिन्",
    // 3 chars
    "एका",
    "एकी",
    "एको",
    "छन्",
    "त्व",
    "थिए",
    "पना",
    "पनि",
    "बाट",
    "भनी",
    "भने",
    "लाई",
    "सँग",
    "सित",
    "हरु",
    "हरू",
    // 2 chars
    "एर",
    "का",
    "की",
    "को",
    "जी",
    "ता",
    "दा",
    "दै",
    "नु",
    "ने",
    "नै",
    "पन",
    "मा",
    "ले",
    // 1 char
    "त",
];

pub fn get_morph_suffix(dev: &str) -> Option<&'static str> {
    MORPH_SUFFIXES
        .iter()
        .copied()
        .find(|&s| dev.ends_with(s) && dev.len() > s.len())
}

/// What the ranker knows about how common a word is.
///
/// Counts are on one scale for every source: occurrences per ~100M tokens of
/// running text (the legacy Nepali corpus had 86M, so its raw counts already
/// are).  0 means the word is unknown.
pub trait WordCounts {
    fn count(&self, word: &str) -> f64;
    /// Frequency-rank percentile in [0, 1]: 0 for the most frequent word,
    /// 1 for an unknown one.
    fn rank_pct(&self, word: &str) -> f64;
}

/// The single-language corpus vocabulary of v1-v7 models.
pub struct VocabCounts<'a> {
    pub freq: &'a HashMap<String, u32>,
    pub ranks: &'a FreqRanks,
}

/// No frequency data: every word is equally unknown. Lets the no-model path
/// share the one ranking function instead of a second scorer.
pub struct NoCounts;

impl WordCounts for NoCounts {
    fn count(&self, _word: &str) -> f64 {
        0.0
    }
    fn rank_pct(&self, _word: &str) -> f64 {
        1.0
    }
}

impl WordCounts for VocabCounts<'_> {
    fn count(&self, word: &str) -> f64 {
        f64::from(self.freq.get(word).copied().unwrap_or(0))
    }
    fn rank_pct(&self, word: &str) -> f64 {
        self.ranks
            .rank_of
            .get(word)
            .copied()
            .unwrap_or(self.ranks.total) as f64
            / self.ranks.total.max(1) as f64
    }
}

/// The per-language lexicon of v8 models: counts in one language, or with no
/// language the highest count in any.
pub struct LexiconCounts<'a> {
    pub lexicon: &'a crate::core::lexicon::Lexicon,
    pub lang: Option<usize>,
}

impl WordCounts for LexiconCounts<'_> {
    fn count(&self, word: &str) -> f64 {
        self.lexicon.count(word, self.lang)
    }
    fn rank_pct(&self, word: &str) -> f64 {
        // Levels are log-frequency, so a falling level is a rising rank.
        1.0 - f64::from(self.lexicon.level(word, self.lang)) / 255.0
    }
}

/// Log frequency, backing off to the stem for an unseen inflected form: an
/// agglutinated word may be absent while its stem is frequent.
pub fn morph_effective_log_freq(dev: &str, counts: &dyn WordCounts) -> f64 {
    let f = counts.count(dev);
    if f > 0.0 {
        return (1.0 + f).ln();
    }
    if let Some(suf) = get_morph_suffix(dev) {
        let stem_f = counts.count(&dev[..dev.len() - suf.len()]);
        if stem_f >= 5.0 {
            return (stem_f.ln() + 3.5).max(0.0);
        }
    }
    0.0
}

#[inline]
pub fn hash_feature(template: u32, arg1: u32, arg2: u32) -> usize {
    let mut x: u64 = ((template as u64) << 48) ^ ((arg1 as u64) << 24) ^ (arg2 as u64);
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    (x as usize) & (HASH_SIZE - 1)
}

#[inline]
pub fn hash_string(s: &str) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for b in s.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}

pub fn extract_dense_features(
    c: &DecodedCandidate,
    rank: usize,
    heur: f64,
    heur_rank: usize,
    roman: &str,
    counts: &dyn WordCounts,
) -> [f64; DENSE_DIM] {
    let f = counts.count(&c.dev);
    let rank_pct = counts.rank_pct(&c.dev);
    let mut feats = [0.0f64; DENSE_DIM];
    feats[0] = c.emit;
    feats[1] = c.lm;
    feats[2] = c.akshara_count as f64;
    feats[3] = rank as f64;
    feats[4] = heur;
    feats[5] = heur_rank as f64;
    feats[6] = (1.0 + f).ln();
    feats[7] = rank_pct;
    feats[8] = if f > 0.0 { 1.0 } else { 0.0 };
    feats[9] = c.dev.chars().count() as f64;
    let matra_total: f64 = MATRAS
        .iter()
        .map(|m| c.dev.matches(*m).count() as f64)
        .sum();
    feats[10] = matra_total;
    for (k, m) in MATRAS.iter().enumerate() {
        feats[11 + k] = c.dev.matches(*m).count() as f64;
    }
    feats[21] = (c.dev.matches('\u{0902}').count() + c.dev.matches('\u{0901}').count()) as f64;
    feats[22] = c.dev.matches('\u{0903}').count() as f64;
    feats[23] = c.dev.matches('\u{094D}').count() as f64;
    let first = c.dev.chars().next();
    feats[24] = first.is_some_and(|ch| "अआइईउऊएऐओऔऋ".contains(ch)) as i32 as f64;
    feats[25] = c.dev.chars().last().is_some_and(|ch| MATRAS.contains(&ch)) as i32 as f64;
    feats[26] = c
        .dev
        .chars()
        .last()
        .is_some_and(|ch| ch == '\u{0902}' || ch == '\u{0901}' || ch == '\u{0903}')
        as i32 as f64;
    feats[27] = roman.chars().count() as f64;
    feats[28] = morph_effective_log_freq(&c.dev, counts);
    feats
}

/// Sparse feature slots of one candidate (language-blind).
pub fn extract_sparse_features(dev: &str, roman: &str, n_ak: usize, aks: &[String]) -> Vec<usize> {
    extract_sparse_features_lang(dev, roman, n_ak, aks, None)
}

/// Sparse feature slots, with a language-specific copy of every feature when
/// the query's language is known: feature augmentation (Daumé III 2007,
/// "Frustratingly Easy Domain Adaptation").  The shared copy learns what
/// holds for all Devanagari; the tagged copy learns what one language does
/// differently (Hindi drops a word-final schwa that Sanskrit writes, Nepali
/// and Hindi spell the same loanword differently, ...).  With `lang = None`
/// the slots are exactly the language-blind ones.
pub fn extract_sparse_features_lang(
    dev: &str,
    roman: &str,
    n_ak: usize,
    aks: &[String],
    lang: Option<usize>,
) -> Vec<usize> {
    let keys = sparse_keys(dev, roman, n_ak, aks);
    let mut feats = Vec::with_capacity(keys.len() * 2);
    for &(template, a, b) in &keys {
        feats.push(hash_feature(template, a, b));
        if let Some(l) = lang {
            // Templates are small integers; the language rides above them.
            feats.push(hash_feature(template | ((l as u32 + 1) << 8), a, b));
        }
    }
    feats.sort_unstable();
    feats.dedup();
    feats
}

/// The lexicalised feature keys of one candidate: (template, arg1, arg2).
fn sparse_keys(dev: &str, roman: &str, n_ak: usize, aks: &[String]) -> Vec<(u32, u32, u32)> {
    let mut feats = Vec::with_capacity(12);
    let dev_chars: Vec<char> = dev.chars().collect();
    let roman_bytes = roman.as_bytes();

    // 1. Length delta bucket
    let delta = (n_ak as i32) - (roman_bytes.len() as i32);
    let delta_bucket = (delta + 10).clamp(0, 20) as u32;
    feats.push((1, delta_bucket, 0));

    // 2. Final akshara x final roman character
    if let (Some(last_ak), Some(&last_r)) = (aks.last(), roman_bytes.last()) {
        let h_ak = hash_string(last_ak);
        feats.push((2, h_ak, last_r as u32));
    }

    // 3. First akshara x first roman character
    if let (Some(first_ak), Some(&first_r)) = (aks.first(), roman_bytes.first()) {
        let h_ak = hash_string(first_ak);
        feats.push((3, h_ak, first_r as u32));
    }

    // 4. Matra x preceding consonant
    for i in 1..dev_chars.len() {
        if MATRAS.contains(&dev_chars[i]) {
            let prev_c = dev_chars[i - 1] as u32;
            let matra_c = dev_chars[i] as u32;
            feats.push((4, prev_c, matra_c));
        }
    }

    // 5. Morphological suffix x roman tail character
    if let Some(suf) = get_morph_suffix(dev) {
        if let Some(&last_r) = roman_bytes.last() {
            let h_suf = hash_string(suf);
            feats.push((5, h_suf, last_r as u32));
        }
    }

    // 6. Final matra x final roman character (W4)
    if let (Some(last_ch), Some(&last_r)) = (dev_chars.last(), roman_bytes.last()) {
        if MATRAS.contains(last_ch) {
            feats.push((6, *last_ch as u32, last_r as u32));
        }
    }

    // 7. Penultimate consonant x final matra
    if dev_chars.len() >= 2 {
        let last_ch = dev_chars[dev_chars.len() - 1];
        let prev_ch = dev_chars[dev_chars.len() - 2];
        if MATRAS.contains(&last_ch) {
            feats.push((7, prev_ch as u32, last_ch as u32));
        }
    }

    feats
}

/// Word-frequency ranks (position in vocabulary sorted by frequency descending).
pub struct FreqRanks {
    pub rank_of: HashMap<String, usize>,
    pub total: usize,
}

/// Vocabulary data needed by the discriminative reranker.
pub struct RerankerData {
    pub freq: HashMap<String, u32>,
    pub ranks: FreqRanks,
}

/// Dense-feature standardisation statistics. Travels in the container
/// (`UnifiedModel::dense_mean` / `dense_std`); no compiled-in fallback.
#[derive(Debug, Clone, Copy)]
pub struct DenseNorm<'a> {
    pub mean: &'a [f64],
    pub std: &'a [f64],
}

static IDENTITY_MEAN: [f64; DENSE_DIM] = [0.0; DENSE_DIM];
static IDENTITY_STD: [f64; DENSE_DIM] = [1.0; DENSE_DIM];

impl DenseNorm<'static> {
    /// Identity standardisation `(x - 0) / 1`.
    pub const fn identity() -> Self {
        Self {
            mean: &IDENTITY_MEAN,
            std: &IDENTITY_STD,
        }
    }
}

impl<'a> DenseNorm<'a> {
    /// The model's own statistics, else identity.
    pub fn from_model(mean: &'a [f64], std: &'a [f64]) -> Self {
        if mean.len() == DENSE_DIM && std.len() == DENSE_DIM {
            Self { mean, std }
        } else {
            Self {
                mean: &IDENTITY_MEAN,
                std: &IDENTITY_STD,
            }
        }
    }

    #[inline]
    fn z(&self, k: usize, x: f64) -> f64 {
        let sd = self.std[k];
        if sd.abs() < 1e-12 {
            0.0
        } else {
            (x - self.mean[k]) / sd
        }
    }
}

impl RerankerData {
    pub fn from_bin_bytes(bytes: &[u8]) -> Option<Self> {
        let freq: HashMap<String, u32> = bincode::deserialize(bytes).ok()?;
        let ranks = FreqRanks::from_freq_map(&freq);
        Some(Self { freq, ranks })
    }
}

impl FreqRanks {
    pub fn from_freq_map(freq: &HashMap<String, u32>) -> Self {
        let mut by_freq: Vec<(&String, u32)> = freq.iter().map(|(w, &c)| (w, c)).collect();
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

/// Rank candidates using the production discriminative model.
pub fn rerank(
    roman: &str,
    candidates: &[DecodedCandidate],
    freq: &HashMap<String, u32>,
    ranks: &FreqRanks,
) -> Vec<(String, f64)> {
    rerank_with_table(roman, candidates, freq, ranks, None, None)
}

/// Candidate ordering and heuristic scores shared by inference and training.
///
/// The trainer used to recompute this inline and got it wrong: it filled
/// `heur_rank` with the identity permutation instead of sorting by `heur`,
/// so dense feature #5 meant "decoder rank" during training and "heuristic
/// rank" at inference.  Both callers now go through this function so the two
/// cannot drift apart again.
///
/// Returns `(order, heur, heur_rank)` where `order` is the candidate list
/// sorted by `emit + lm`, `heur[i]` is the baseline heuristic cost of
/// `order[i]`, and `heur_rank[i]` is that candidate's 0-based rank once
/// sorted by `heur` ascending.
pub fn rank_candidates<'a>(
    candidates: &'a [DecodedCandidate],
    counts: &dyn WordCounts,
) -> (Vec<&'a DecodedCandidate>, Vec<f64>, Vec<usize>) {
    let mut order: Vec<&DecodedCandidate> = candidates.iter().collect();
    order.sort_by(|a, b| {
        (a.emit + a.lm)
            .total_cmp(&(b.emit + b.lm))
            .then(a.dev.cmp(&b.dev))
    });

    let heur: Vec<f64> = order
        .iter()
        .map(|c| c.emit + LM_W * c.lm - VOCAB_W * (1.0 + counts.count(&c.dev)).ln())
        .collect();

    let mut heur_order: Vec<usize> = (0..order.len()).collect();
    heur_order.sort_by(|a, b| {
        heur[*a]
            .total_cmp(&heur[*b])
            .then(order[*a].dev.cmp(&order[*b].dev))
    });
    let mut heur_rank = vec![0usize; order.len()];
    for (r, i) in heur_order.into_iter().enumerate() {
        heur_rank[i] = r;
    }

    (order, heur, heur_rank)
}

/// Rank candidates with an optional custom sparse weight table (from UnifiedModel).
pub fn rerank_with_table(
    roman: &str,
    candidates: &[DecodedCandidate],
    freq: &HashMap<String, u32>,
    ranks: &FreqRanks,
    custom_sparse_table: Option<&[i8]>,
    custom_sparse_scale: Option<f64>,
) -> Vec<(String, f64)> {
    rerank_with_norm(
        roman,
        candidates,
        &VocabCounts { freq, ranks },
        custom_sparse_table,
        custom_sparse_scale,
        DenseNorm::identity(),
        None,
        None,
        None,
    )
}

/// Language conditioning for one query: which language the user is typing
/// (an index into the model's `langs`) and that language's dense weight
/// offsets.  `None` everywhere means language-blind ranking.
#[derive(Debug, Clone, Copy)]
pub struct LangCond<'a> {
    pub index: usize,
    pub dense: &'a [f64],
}

/// As `rerank_with_table`, with explicit dense-feature normalisation statistics
/// and optionally jointly-trained dense weights (v6+ containers).
///
/// When `custom_dense_weights` is `Some`, those 29 weights are used in place of
/// the compiled-in `W_DENSE` constant.  `None` falls back to `W_DENSE` exactly
/// as before, so v5 and earlier containers are unaffected.
#[allow(clippy::too_many_arguments)]
pub fn rerank_with_norm(
    roman: &str,
    candidates: &[DecodedCandidate],
    counts: &dyn WordCounts,
    custom_sparse_table: Option<&[i8]>,
    custom_sparse_scale: Option<f64>,
    norm: DenseNorm<'_>,
    custom_dense_weights: Option<&[f64]>,
    lang: Option<LangCond<'_>>,
    gamma: Option<f64>,
) -> Vec<(String, f64)> {
    if candidates.is_empty() {
        return vec![];
    }

    let (order, heur, heur_rank) = rank_candidates(candidates, counts);

    // Raw decoder order: `order` is already sorted by emit + lm, so emitting it
    // with descending scores reproduces the generative ranking exactly.
    if crate::core::ablation::no_rerank() {
        return order
            .iter()
            .enumerate()
            .map(|(i, c)| (c.dev.clone(), -(i as f64)))
            .collect();
    }

    let n = order.len();
    let mut scores: Vec<f64> = Vec::with_capacity(n);
    let mut heur_std: Vec<f64> = Vec::with_capacity(n);

    // Cascade: full feature extraction (akshara segmentation, matra scans,
    // sparse hashing) costs far more than the heuristic, and candidates the
    // heuristic already ranks far down never reach the top of the final list.
    // Score only the most promising `RERANK_FULL_DEPTH` in full and let the
    // rest keep their heuristic ordering, offset below the scored block so the
    // two halves cannot interleave.
    let full_depth = rerank_full_depth().min(n);
    let mut worst_scored = f64::INFINITY;

    for (i, c) in order.iter().enumerate() {
        if heur_rank[i] >= full_depth {
            // Not in the cascade: placeholder, filled in after the scored pass.
            scores.push(f64::NAN);
            heur_std.push(norm.z(4, heur[i]));
            continue;
        }
        let dense = extract_dense_features(c, i, heur[i], heur_rank[i], roman, counts);
        let aks = akshara::segment(&c.dev);
        let sparse = extract_sparse_features_lang(
            &c.dev,
            roman,
            c.akshara_count,
            &aks,
            lang.map(|l| l.index),
        );

        let mut s = 0.0f64;
        let w_dense: &[f64] = match custom_dense_weights {
            Some(w) if w.len() >= DENSE_DIM => w,
            _ => &[],
        };
        for (k, &xk) in dense.iter().enumerate() {
            let wk = w_dense.get(k).copied().unwrap_or(0.0)
                + lang.map_or(0.0, |l| l.dense.get(k).copied().unwrap_or(0.0));
            // Exp 6: a NaN weight or feature must not poison the whole list.
            if wk.is_finite() {
                let z = norm.z(k, xk);
                if z.is_finite() {
                    s += wk * z;
                }
            }
        }
        for &h in sparse
            .iter()
            .filter(|_| !crate::core::ablation::no_sparse())
        {
            let (b, scale) = match custom_sparse_table {
                Some(t) if h < t.len() => (
                    t[h],
                    custom_sparse_scale
                        .filter(|s| s.is_finite() && *s >= 0.0)
                        .unwrap_or(1.0),
                ),
                _ => (0, 1.0),
            };
            s += (b as f64) * scale;
        }
        worst_scored = worst_scored.min(s);
        scores.push(s);
        heur_std.push(norm.z(4, heur[i]));
    }

    // Place unscored candidates below every scored one, ordered by heuristic
    // rank, so the cascade only ever affects the tail of the list.
    //
    // Note: these synthetic scores do participate in the GAMMA z-blend below,
    // which shifts the score vector's mean and spread relative to scoring every
    // candidate.  The ordering among *scored* candidates survives (z-scoring is
    // affine), but the blend weight against the heuristic moves slightly with
    // the scored/unscored ratio.  Measured on the 2,108-case AK-Freq split:
    // depth 24 -> 81.83/92.22, depth 50 (no cascade) -> 81.69/92.22, depth 12
    // -> 81.78/92.41, depth 6 -> 81.31/91.98.  The effect is within noise down
    // to depth 12; below that the cascade starts dropping real candidates.
    if worst_scored.is_finite() {
        for (i, sc) in scores.iter_mut().enumerate() {
            if sc.is_nan() {
                *sc = worst_scored - 1.0 - (heur_rank[i] as f64) * 1e-6;
            }
        }
    } else {
        for (i, sc) in scores.iter_mut().enumerate() {
            *sc = -(heur_rank[i] as f64);
        }
    }

    // Blend weight: the AKSHAR_GAMMA experiment override, else the model's
    // calibrated value for this query, else the compiled default.  A NaN
    // (e.g. AKSHAR_GAMMA=NaN) fails every comparison and would poison the
    // blend, so only finite values count.
    let gamma = crate::core::ablation::gamma()
        .or(gamma)
        .filter(|g| g.is_finite())
        .unwrap_or(GAMMA);
    if gamma >= 1.0 {
        let mut blended: Vec<(String, f64)> = order
            .iter()
            .zip(scores)
            .map(|(c, s)| (c.dev.clone(), s))
            .collect();
        blended.sort_by(|a, b| b.1.total_cmp(&a.1));
        blended
    } else if gamma <= 0.0 {
        let mut blended: Vec<(String, f64)> = order
            .iter()
            .zip(heur)
            .map(|(c, h)| (c.dev.clone(), -h))
            .collect();
        blended.sort_by(|a, b| b.1.total_cmp(&a.1));
        blended
    } else {
        zscore(&mut heur_std);
        zscore(&mut scores);
        let mut blended: Vec<(String, f64)> = order
            .iter()
            .zip(scores)
            .zip(heur_std)
            .map(|((c, s), zh)| (c.dev.clone(), (1.0 - gamma) * (-zh) + gamma * s))
            .collect();
        blended.sort_by(|a, b| b.1.total_cmp(&a.1));
        blended
    }
}

// ---------------------------------------------------------------------------
// Feature names
// ---------------------------------------------------------------------------

/// Dense feature names, in `extract_dense_features` order.
pub fn feature_names() -> [&'static str; DENSE_DIM] {
    [
        "emit",
        "lm",
        "akshara_count",
        "decoder_rank",
        "heuristic",
        "heuristic_rank",
        "log1p_freq",
        "freq_rank_pct",
        "in_vocab",
        "len_dev",
        "matra_total",
        "matra_a",
        "matra_i",
        "matra_ii",
        "matra_u",
        "matra_uu",
        "matra_e",
        "matra_ai",
        "matra_o",
        "matra_au",
        "matra_ri",
        "nasals",
        "visarga",
        "halants",
        "vowel_initial",
        "ends_matra",
        "ends_nasal_visarga",
        "len_roman",
        "morph_effective_log_freq",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morph_table_is_longest_first() {
        assert_eq!(MORPH_SUFFIXES.len(), 34);
        let mut prev = usize::MAX;
        for &s in MORPH_SUFFIXES.iter() {
            let n = s.chars().count();
            assert!(n <= prev, "table not longest-first at {s}");
            assert!(n > 0);
            prev = n;
        }
    }

    #[test]
    fn longest_match_wins_over_prefix() {
        // "आएको" ends in both "एको" (len 3) and "को" (len 2): longest wins.
        assert_eq!(get_morph_suffix("आएको"), Some("एको"));
        assert_eq!(get_morph_suffix("गरेको"), Some("को"));
        assert_eq!(get_morph_suffix("घरबाट"), Some("बाट"));
        assert_eq!(get_morph_suffix("साथीहरू"), Some("हरू"));
        // The whole-word rule leaves at least one character:
        assert_eq!(get_morph_suffix("बाट"), None);
        assert_eq!(get_morph_suffix("त"), None);
    }
}
