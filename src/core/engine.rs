// File: src/core/engine.rs
//
// IME engine: one decoder, one coherent evidence score.
//
// Candidates come from four sources, each contributing an evidence score:
//
//   * the generative decoder (fresh transliterations of the roman prefix),
//   * the corpus roman->devanagari lexicon (exact + prefix completions),
//   * the user's learned dictionary (trie + fuzzy SymSpell),
//   * the context model (re-ranks words the user has typed before).
//
// The final score is the max evidence across sources: a word confirmed as a
// real word outranks a merely-transliterated form, and user-confirmed words
// climb as their frequency grows.

use crate::core::{
    context::ContextModel,
    decoder::{DecoderConfig, ModelDecoder},
    lexicon::RomanLexicon,
    normalizer::expand_query_variants,
    reranker::{Reranker, NUM_FEATURES},
    translit_model::TranslitModel,
    trie::Trie,
    types::{TransliterationModel, WordId},
};
use crate::fuzzy::symspell::SymSpell;
use crate::learning::{LearningEngine, WordConfirmation};
#[cfg(not(target_arch = "wasm32"))]
use crate::persistence::{load_from_disk, save_to_disk};
use std::collections::HashMap;
use std::path::PathBuf;

const CONTEXT_WINDOW_SIZE: usize = 3;
const MAX_EDIT_DISTANCE: usize = 2;
const QUERY_VARIANT_LIMIT: usize = 6;

/// Decoder beam for the IME (accuracy/speed sweet spot, see M2 eval).
const DECODER_BEAM: usize = 64;
/// Scale converting a reranker log-score into the engine's higher-better u64 score.
const FRESH_SCALE: f64 = 800_000.0;
/// Purnabiram (।, U+0964) — mapped from a trailing '.'.
const PURNABIRAM: char = '\u{0964}';
/// Weight on the corpus-bigram context term, in nats.
///
/// The term is a shrunk pointwise mutual information, `ln P(w | prev) - ln P(w)`
/// scaled by `f / (f + k)`, added to the reranker's log-score *before* it is
/// squashed onto the u64 scale.  PMI rather than raw `ln P(w | prev)` so that
/// candidates the table does not cover are neither rewarded nor punished: the
/// term is 0 exactly when the previous word says nothing about this candidate.
///
/// 0.10 is the measured optimum on held-out running text (evaluate_sentences,
/// 1500 sentences / 40,480 words):
///
///   context off        word@1 89.69%   sentence-exact 26.73%
///   weight 0.10        word@1 89.85%   sentence-exact 26.93%
///   weight 0.25        word@1 89.78%   sentence-exact 27.07%
///   weight 0.50        word@1 89.49%   sentence-exact 26.33%
///   weight 1.00        word@1 88.94%   sentence-exact 25.20%
///
/// Note what that curve says: the whole corpus-bigram table is worth about
/// +0.16pp of word@1.  Three different fusion designs — the original
/// post-squash `40_000 * ln(1+f)` bonus, unshrunk PMI, and this one — all land
/// on the same ceiling, so the limit is the table, not the arithmetic.  At
/// 19.54 MB that is ~122 MB per accuracy point, against 0.4 MB/point for the
/// sparse reranker and 7 MB/point for the syllable trigram LM.  It is the
/// first thing to drop for a size-constrained build.
///
/// A plausible cause is that the table is built with `--bigram-min-freq 5`,
/// which preferentially removes pairs involving the rarer of two spelling
/// variants — exactly the entries needed to settle the dominant error class
/// (बिच vs बीच).  Worth testing before concluding the table is inherently weak.
const DEFAULT_BIGRAM_WEIGHT: f64 = 0.10;
/// Count discount `k` in the PMI shrinkage factor `f / (f + k)`.
const DEFAULT_BIGRAM_SHRINK: f64 = 10.0;
/// Bonus for a candidate that is both decoder-generated AND an exact corpus
/// word.  Additive on top of the fresh score so the decoder's ranking among
/// corpus words is preserved (a flat absolute score made arbitrary-order
/// lexicon entries outrank better decoder candidates; E0 measured -2.1 pts).
const LEXICON_EXACT_BONUS: u64 = 0;
/// A corpus word the decoder did not generate itself.
const LEXICON_ONLY_SCORE: u64 = 5_000;
/// User-confirmed word from the learned trie.  Above any fresh+lexicon score
/// so a word the user picked before always wins.
const USER_TRIE_BASE: u64 = 500_000;
/// Fuzzy (edit-distance) match over user-learned roman variants.
const FUZZY_BASE: u64 = 50_000;
const FUZZY_DISTANCE_PENALTY_SCALE: u64 = 12_000;

pub struct ImeEngine {
    pub decoder: ModelDecoder,
    pub reranker: Reranker,
    pub lexicon: Option<RomanLexicon>,
    /// Corpus-vocabulary data for the discriminative reranker (native builds with
    /// word_freq_text.bin present; None on WASM-lite or missing file).
    reranker_data: Option<crate::core::reranker::RerankerData>,
    /// W3: Candidate Union word trie over vocabulary.
    word_trie: Option<crate::core::wordtrie::WordTrie>,
    /// E6: corpus word-bigram table (data/word_bigrams.bin) for
    /// context-conditioned reranking; None on WASM-lite or missing file.
    corpus_bigrams: Option<HashMap<String, Vec<(String, u32)>>>,
    /// Whether the corpus-bigram context boost is applied (harness A/B).
    bigram_context_enabled: bool,
    /// Per-ln-unit boost applied to candidates forming a bigram with the
    /// previously committed word. Tunable via set_bigram_weight.
    bigram_weight: f64,
    /// Count discount for the PMI term; see DEFAULT_BIGRAM_SHRINK.
    bigram_shrink: f64,
    /// Total corpus token count, denominator for the unigram P(w) in the PMI
    /// context term.  Cached because it is a sum over the whole vocabulary.
    vocab_total: f64,
    /// Last word the user committed (drives the corpus-bigram context).
    last_word: Option<String>,
    pub trie: Trie,
    pub context_model: ContextModel,
    pub symspell: SymSpell,
    pub(crate) transliteration_model: TransliterationModel,
    learning_engine: LearningEngine,
    #[allow(dead_code)]
    dictionary_path: Option<String>,
    pub sparse_table: Option<Vec<i8>>,
}

impl ImeEngine {
    pub fn new() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let unified_path = data_path("akshar.model");
            if unified_path.exists() {
                if let Ok(unified) = crate::core::unified::UnifiedModel::load(&unified_path) {
                    return Self::from_unified(unified);
                }
            }
        }
        let model = load_model_or_default();
        let decoder = ModelDecoder::with_config(
            model,
            DecoderConfig {
                beam_width: DECODER_BEAM,
                ..DecoderConfig::default()
            },
        );
        let lexicon = load_lexicon();
        let reranker = load_reranker(lexicon.clone());
        let reranker_data = load_reranker_data();
        let word_trie = reranker_data.as_ref().map(|v| {
            crate::core::wordtrie::WordTrie::from_freq_map(&v.freq, &|a| decoder.model.akshara_id(a), 1)
        });
        let vocab_total = total_tokens(reranker_data.as_ref());
        Self {
            decoder,
            reranker,
            lexicon,
            reranker_data,
            word_trie,
            corpus_bigrams: load_bigrams(),
            bigram_context_enabled: true,
            bigram_weight: DEFAULT_BIGRAM_WEIGHT,
            bigram_shrink: DEFAULT_BIGRAM_SHRINK,
            vocab_total,
            last_word: None,
            trie: Trie::new(),
            context_model: ContextModel::new(CONTEXT_WINDOW_SIZE),
            symspell: SymSpell::default(),
            transliteration_model: TransliterationModel::new(),
            learning_engine: LearningEngine::new(),
            dictionary_path: None,
            sparse_table: None,
        }
    }

    /// Construct engine directly from a unified model container.
    pub fn from_unified(mut unified: crate::core::unified::UnifiedModel) -> Self {
        unified.translit.build_trigram_index();
        let decoder = ModelDecoder::with_config(
            unified.translit,
            DecoderConfig {
                beam_width: DECODER_BEAM,
                ..DecoderConfig::default()
            },
        );
        let lexicon = None;
        let reranker = load_reranker(lexicon.clone()).with_freq(Some(unified.vocab_freq.clone()));
        let ranks = crate::core::reranker::FreqRanks::from_freq_map(&unified.vocab_freq);
        let word_trie = Some(crate::core::wordtrie::WordTrie::from_freq_map(
            &unified.vocab_freq,
            &|a| decoder.model.akshara_id(a),
            1,
        ));
        let vocab_total = unified.vocab_freq.values().map(|&f| f as f64).sum();
        let reranker_data = Some(crate::core::reranker::RerankerData {
            freq: unified.vocab_freq,
            ranks,
        });
        Self {
            decoder,
            reranker,
            lexicon,
            reranker_data,
            word_trie,
            corpus_bigrams: unified.bigrams,
            bigram_context_enabled: true,
            bigram_weight: DEFAULT_BIGRAM_WEIGHT,
            bigram_shrink: DEFAULT_BIGRAM_SHRINK,
            vocab_total,
            last_word: None,
            trie: Trie::new(),
            context_model: ContextModel::new(CONTEXT_WINDOW_SIZE),
            symspell: SymSpell::default(),
            transliteration_model: TransliterationModel::new(),
            learning_engine: LearningEngine::new(),
            dictionary_path: None,
            sparse_table: if unified.sparse_reranker_table.is_empty() {
                None
            } else {
                Some(unified.sparse_reranker_table)
            },
        }
    }

    /// Load engine from a unified model file (`akshar.model`).
    pub fn from_unified_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let unified = crate::core::unified::UnifiedModel::load(path)?;
        Ok(Self::from_unified(unified))
    }

    /// Load engine from in-memory unified model bytes.
    pub fn from_unified_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let unified = crate::core::unified::UnifiedModel::from_bytes(bytes)?;
        Ok(Self::from_unified(unified))
    }

    pub fn from_file_or_new(path: &str) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut engine = load_from_disk(std::path::Path::new(path)).unwrap_or_else(|_| Self::new());
            engine.dictionary_path = Some(path.to_string());
            engine
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = path;
            Self::new()
        }
    }

    /// Create engine from raw model bytes (no filesystem).
    pub fn from_bytes(
        model_bytes: &[u8],
        lexicon_bytes: Option<&[u8]>,
        reranker_json: Option<&str>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_bytes_with_weights(model_bytes, lexicon_bytes, reranker_json)
    }

    /// Build from bytes, allowing caller to supply already-parsed model.
    pub fn from_model(model: TranslitModel, lexicon: Option<RomanLexicon>, reranker: Option<Reranker>) -> Self {
        let decoder = ModelDecoder::with_config(
            model,
            DecoderConfig {
                beam_width: DECODER_BEAM,
                ..DecoderConfig::default()
            },
        );
        let reranker = reranker.unwrap_or_else(|| load_reranker(lexicon.clone()));
        let reranker_data = load_reranker_data();
        let word_trie = reranker_data.as_ref().map(|v| {
            crate::core::wordtrie::WordTrie::from_freq_map(&v.freq, &|a| decoder.model.akshara_id(a), 1)
        });
        let vocab_total = total_tokens(reranker_data.as_ref());
        Self {
            decoder,
            reranker,
            lexicon,
            reranker_data,
            word_trie,
            corpus_bigrams: load_bigrams(),
            bigram_context_enabled: true,
            bigram_weight: DEFAULT_BIGRAM_WEIGHT,
            bigram_shrink: DEFAULT_BIGRAM_SHRINK,
            vocab_total,
            last_word: None,
            trie: Trie::new(),
            context_model: ContextModel::new(CONTEXT_WINDOW_SIZE),
            symspell: SymSpell::new(MAX_EDIT_DISTANCE),
            transliteration_model: HashMap::new(),
            learning_engine: LearningEngine::new(),
            dictionary_path: None,
            sparse_table: None,
        }
    }

    pub fn from_bytes_with_weights(
        model_bytes: &[u8],
        lexicon_bytes: Option<&[u8]>,
        reranker_json: Option<&str>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut model = TranslitModel::from_bytes(model_bytes)?;
        if !model.validate() {
            return Err("invalid translit model".into());
        }
        // from_bytes already builds trigram index
        let _ = &mut model; // keep
        let lexicon = if let Some(b) = lexicon_bytes {
            if b.is_empty() { None } else { RomanLexicon::from_bytes(b).ok() }
        } else { None };
        let reranker = if let Some(json) = reranker_json {
            Self::parse_reranker_json(json, lexicon.clone())
        } else {
            load_reranker(lexicon.clone())
        };
        Ok(Self::from_model(model, lexicon, Some(reranker)))
    }

    fn parse_reranker_json(json_str: &str, lexicon: Option<RomanLexicon>) -> Reranker {
        let mut weights = [1.0f64; NUM_FEATURES];
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
            if let Some(w) = v.get("weights") {
                let names = crate::core::reranker::feature_names();
                for (i, name) in names.iter().enumerate() {
                    if let Some(val) = w.get(*name).and_then(|x| x.as_f64()) {
                        weights[i] = val;
                    }
                }
            } else if let Ok(arr) = serde_json::from_str::<[f64; NUM_FEATURES]>(json_str) {
                weights = arr;
            }
        }
        Reranker::new(weights, lexicon)
    }

    /// Serialise learned state (trie + context + symspell) to bytes for persistence.
    pub fn learned_state_to_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let state = crate::persistence::SerializableState::from_engine(self);
        state.to_bytes()
    }

    /// Load learned state from bytes (e.g. from localStorage).
    pub fn load_learned_state_from_bytes(&mut self, bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        let state = crate::persistence::SerializableState::from_bytes(bytes)?;
        state.apply_to_engine(self);
        Ok(())
    }

    /// Entry point for every runtime (IBus C layer, WASM): applies the pure
    /// mappings (ASCII digits → Devanagari digits, trailing '.' → purnabiram)
    /// before the roman goes through the statistical model.
    pub fn get_suggestions(&self, prefix: &str, count: usize) -> Vec<(String, u64)> {
        if prefix.is_empty() {
            return vec![];
        }
        let count = count.max(1);

        // Purnabiram (।): a trailing '.' asks for it on the result; a lone
        // '.' *is* purnabiram.
        let (base, purnabiram) = match prefix.strip_suffix('.') {
            Some("") => return vec![(PURNABIRAM.to_string(), FRESH_SCALE as u64)],
            Some(b) => (b, true),
            None => (prefix, false),
        };
        let finish = |mut out: Vec<(String, u64)>| {
            if purnabiram {
                for (s, _) in out.iter_mut() {
                    s.push(PURNABIRAM);
                }
            }
            out
        };

        // Split the input into letter runs and digit runs. Digits never enter
        // the model — they map to Devanagari digits directly.
        let mut segments: Vec<Result<String, String>> = Vec::new(); // Ok(letters) / Err(digits)
        for part in base.split_inclusive(|c: char| c.is_ascii_digit()) {
            if part.is_empty() {
                continue;
            }
            if part.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                let last = segments.last_mut();
                match last {
                    Some(Err(d)) => d.push_str(part),
                    _ => segments.push(Err(part.to_string())),
                }
            } else {
                let mut digits_end = part.len();
                for (i, c) in part.char_indices() {
                    if c.is_ascii_digit() {
                        digits_end = i;
                        break;
                    }
                }
                let (letters, trailing_digits) = part.split_at(digits_end);
                segments.push(Ok(letters.to_string()));
                if !trailing_digits.is_empty() {
                    segments.push(Err(trailing_digits.to_string()));
                }
            }
        }
        let map_digits = |d: &str| -> String {
            d.chars()
                .map(|c| {
                    if c.is_ascii_digit() {
                        char::from_u32('\u{0966}' as u32 + c as u32 - '0' as u32)
                            .unwrap_or(c)
                    } else {
                        c
                    }
                })
                .collect()
        };

        let letter_runs: Vec<&String> = segments.iter().filter_map(|s| s.as_ref().ok()).collect();
        if letter_runs.is_empty() {
            // Pure number: 123 -> १२३ (no model involved).
            let digits = segments
                .iter()
                .filter_map(|s| s.as_ref().err())
                .cloned()
                .collect::<Vec<_>>()
                .join("");
            return finish(vec![(map_digits(&digits), FRESH_SCALE as u64)]);
        }
        if letter_runs.len() == 1 {
            // Single word with optional leading/trailing digits: decode the
            // word, then place the mapped digit runs at their original edges.
            let roman = letter_runs[0].as_str();
            let lead_digits = match segments.first() {
                Some(Err(d)) => map_digits(d),
                _ => String::new(),
            };
            let trail_digits = match segments.last() {
                Some(Err(d)) => map_digits(d),
                _ => String::new(),
            };
            let mut out = self.suggestions_for_roman(roman, count);
            for (s, _) in out.iter_mut() {
                *s = format!("{lead_digits}{s}{trail_digits}");
            }
            return finish(out);
        }
        // Digits between words ("ka2024ma"): decode each letter run to its
        // top choice and interleave the mapped digits — one combined result.
        let mut combined = String::new();
        for seg in &segments {
            match seg {
                Ok(roman) => {
                    if let Some((top, _)) = self.suggestions_for_roman(roman, 1).first() {
                        combined.push_str(top);
                    }
                }
                Err(d) => combined.push_str(&map_digits(d)),
            }
        }
        finish(vec![(combined, FRESH_SCALE as u64)])
    }

    /// The statistical path: roman (letters only) -> ranked Devanagari words.
    fn suggestions_for_roman(&self, prefix: &str, count: usize) -> Vec<(String, u64)> {
        if prefix.is_empty() {
            return vec![];
        }
        let query_variants = expand_query_variants(prefix, QUERY_VARIANT_LIMIT);

        let mut candidates: HashMap<String, u64> = HashMap::new();
        let mut add = |dev: String, score: u64| {
            candidates
                .entry(dev)
                .and_modify(|s| *s = (*s).max(score))
                .or_insert(score);
        };

        // 1. Fresh transliterations from the generative decoder.  With the
        //    corpus vocabulary available, rank them with the trained discriminative
        //    reranker (81.02% native top-1 on test); otherwise fall back to
        //    the 5-feature MERT reranker.  We decode the base roman only: the
        //    model's learned emissions already absorb v/w and vowel-length
        //    spelling variants, so re-decoding soft variants is pure latency.
        //    Depth 50 matches the depth the discriminative model was trained on.
        let mut fresh_scores: HashMap<String, u64> = HashMap::new();
        if let Some(qv) = query_variants.first() {
            let roman = qv.roman.as_str();
            let cands = self.decoder.decode_union(roman, (count * 4).max(50), self.word_trie.as_ref());
            let ranked: Vec<(String, f64)> = match &self.reranker_data {
                Some(data) => crate::core::reranker::rerank_with_table(
                    roman,
                    &cands,
                    &data.freq,
                    &data.ranks,
                    self.sparse_table.as_deref(),
                ),
                None => self.reranker.rerank(roman, cands),
            };
            // Apply the corpus-bigram context in the reranker's own log space,
            // then convert to the engine's higher-better u64 scale.  Doing it
            // in this order is what makes the context term behave like
            // evidence: after the hyperbolic squash below, equal amounts of
            // evidence would move candidates by wildly unequal amounts.
            let mut ranked: Vec<(String, f64)> = ranked
                .into_iter()
                .map(|(dev, s)| {
                    let s = s + self.bigram_weight * self.bigram_pmi(&dev);
                    (dev, s)
                })
                .collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

            let s_max = ranked.first().map(|(_, s)| *s).unwrap_or(0.0);
            for (dev, s) in ranked {
                let cost = (s_max - s).max(0.0);
                let score = (FRESH_SCALE / (1.0 + cost)).round().max(1.0) as u64;
                fresh_scores.insert(dev.clone(), score);
                add(dev, score);
            }
        }

        for qv in &query_variants {
            let roman = qv.roman.as_str();

            // 2. Lexicon evidence: only EXACT roman matches (a confirmed word).
            //    The Aksharantar lexicon is mined from news and mostly holds
            //    long compound words, so prefix matching would flood the list
            //    with compounds like नेपालअधिराज्य; the decoder already covers
            //    prefix transliteration.  The bonus is additive on the fresh
            //    score when the decoder also produced the word, so decoder
            //    ranking survives; corpus-only words get a standalone score.
            if let Some(lx) = &self.lexicon {
                for dev in lx.lookup_exact(roman) {
                    let bonus = LEXICON_EXACT_BONUS.saturating_sub(qv.penalty);
                    match fresh_scores.get(&dev) {
                        Some(s) => add(dev, s.saturating_add(bonus)),
                        None => add(dev, LEXICON_ONLY_SCORE.saturating_sub(qv.penalty)),
                    }
                }
            }

            // 3. User-learned dictionary (trie).
            for (word_id, freq) in self.trie.get_top_k_suggestions(roman, count * 3) {
                if let Some(meta) = self.trie.metadata_store.get(word_id) {
                    add(meta.devanagari.clone(), USER_TRIE_BASE.saturating_add(freq));
                }
            }

            // 4. Fuzzy matches over user-learned roman variants (typo tolerance).
            for word_id in self.symspell.lookup(roman) {
                if let Some(meta) = self.trie.metadata_store.get(word_id) {
                    if let Some(min_dist) = self.min_roman_distance(roman, meta, MAX_EDIT_DISTANCE)
                    {
                        let dist_penalty = (min_dist as u64) * FUZZY_DISTANCE_PENALTY_SCALE;
                        let score = FUZZY_BASE.saturating_sub(dist_penalty);
                        add(meta.devanagari.clone(), score);
                    }
                }
            }
        }

        // The corpus-bigram context is applied in step 1, inside the reranker's
        // log space, rather than as a post-hoc bonus on the squashed score.

        // 5. Context re-rank for words the user has typed before.
        let mut with_ids: Vec<(WordId, u64)> = candidates
            .iter()
            .filter_map(|(dev, score)| {
                self.trie
                    .find_word_id_by_devanagari(dev)
                    .map(|id| (id, *score))
            })
            .collect();
        self.context_model.rerank_suggestions(&mut with_ids);
        for (id, new_score) in with_ids {
            if let Some(dev) = self.trie.metadata_store.get(id).map(|m| &m.devanagari) {
                if let Some(entry) = candidates.get_mut(dev) {
                    *entry = new_score;
                }
            }
        }

        let mut out: Vec<(String, u64)> = candidates.into_iter().collect();
        out.sort_by_key(|&(_, score)| std::cmp::Reverse(score));
        out.truncate(count);
        out
    }

    /// E6 harness controls: enable/disable and tune the corpus-bigram
    /// context boost (A/B measurement via evaluate_context).
    pub fn set_bigram_context_enabled(&mut self, on: bool) {
        self.bigram_context_enabled = on;
    }

    /// Weight on the PMI context term, in nats.  1.0 means "trust the corpus
    /// bigram exactly as much as the reranker's own log-score".
    pub fn set_bigram_weight(&mut self, weight: f64) {
        self.bigram_weight = weight;
    }

    /// Count discount `k` in the PMI shrinkage factor `f / (f + k)`.
    pub fn set_bigram_shrink(&mut self, k: f64) {
        self.bigram_shrink = k;
    }

    /// Pointwise mutual information between the previously committed word and
    /// a candidate: `ln P(w | prev) - ln P(w)`, in nats.
    ///
    /// Returns 0.0 when there is no context, no bigram table, no entry for
    /// `prev`, or no entry for `(prev, w)` — i.e. whenever the context carries
    /// no information about this candidate.  That neutrality is the point: an
    /// absolute `ln P(w | prev)` would systematically punish every candidate
    /// the table happens not to cover, which is most of them.
    ///
    /// The result is added to the reranker's log-score before it is squashed
    /// onto the u64 evidence scale, so it composes with the reranker's own
    /// log-linear score instead of fighting a nonlinear transform.
    fn bigram_pmi(&self, dev: &str) -> f64 {
        if !self.bigram_context_enabled || self.vocab_total <= 0.0 {
            return 0.0;
        }
        let (Some(prev), Some(bigrams)) = (&self.last_word, &self.corpus_bigrams) else {
            return 0.0;
        };
        let Some(succ) = bigrams.get(prev) else {
            return 0.0;
        };
        // Successor lists are stored frequency-sorted, not key-sorted, so this
        // is a linear scan.  Lists average ~14 entries; the compact model
        // format will make this a binary search.
        let Some(&(_, joint)) = succ.iter().find(|(w, _)| w == dev) else {
            return 0.0;
        };
        let context_total: f64 = succ.iter().map(|(_, f)| *f as f64).sum();
        if context_total <= 0.0 {
            return 0.0;
        }
        let Some(data) = &self.reranker_data else {
            return 0.0;
        };
        let unigram = data.freq.get(dev).copied().unwrap_or(0) as f64;
        if unigram <= 0.0 {
            return 0.0;
        }
        let p_cond = joint as f64 / context_total;
        let p_uni = unigram / self.vocab_total;
        let pmi = (p_cond / p_uni).ln();

        // Shrink towards 0 by the joint count.  Raw PMI is dominated by its
        // low-count tail: a pair seen 5 times involving a rare word yields a
        // huge score and promotes that rare word over the correct frequent
        // one.  The f/(f+k) factor is the standard discount — it leaves
        // well-attested pairs almost untouched and suppresses the noise.
        let f = joint as f64;
        pmi * (f / (f + self.bigram_shrink))
    }

    /// Set the preceding-word context without learning the word.
    ///
    /// `user_confirms` both sets the context *and* teaches the word to the
    /// user trie / SymSpell / adaptive model.  Offline harnesses that walk a
    /// gold sentence must not do the latter: learning the gold word makes
    /// every later occurrence trivially correct and the measurement
    /// self-fulfilling.  This is the context half on its own.
    pub fn set_context_word(&mut self, devanagari: &str) {
        self.last_word = if devanagari.is_empty() {
            None
        } else {
            Some(devanagari.to_string())
        };
    }

    pub fn user_confirms(&mut self, roman: &str, devanagari: &str) {
        if roman.is_empty() || devanagari.is_empty() {
            return;
        }
        self.last_word = Some(devanagari.to_string());
        let confirmation = WordConfirmation {
            roman: roman.to_string(),
            devanagari: devanagari.to_string(),
        };
        self.learning_engine.learn(
            &mut self.trie,
            &mut self.context_model,
            &mut self.symspell,
            &mut self.transliteration_model,
            &confirmation,
        );
    }

    fn min_roman_distance(
        &self,
        roman_query: &str,
        metadata: &crate::core::types::WordMetadata,
        max_distance: usize,
    ) -> Option<usize> {
        metadata
            .variants
            .iter()
            .filter_map(|variant| {
                let raw = Self::bounded_levenshtein(roman_query, variant, max_distance);
                let collapsed_query = Self::collapse_vowel_runs(roman_query);
                let collapsed_variant = Self::collapse_vowel_runs(variant);
                let collapsed =
                    Self::bounded_levenshtein(&collapsed_query, &collapsed_variant, max_distance);
                match (raw, collapsed) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                }
            })
            .min()
    }

    fn bounded_levenshtein(a: &str, b: &str, max_distance: usize) -> Option<usize> {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        if a_chars.len().abs_diff(b_chars.len()) > max_distance {
            return None;
        }
        let mut prev: Vec<usize> = (0..=b_chars.len()).collect();
        let mut curr = vec![0usize; b_chars.len() + 1];
        for (i, ca) in a_chars.iter().enumerate() {
            curr[0] = i + 1;
            let mut row_min = curr[0];
            for (j, cb) in b_chars.iter().enumerate() {
                let replace_cost = if ca == cb { 0 } else { 1 };
                let deletion = prev[j + 1] + 1;
                let insertion = curr[j] + 1;
                let replacement = prev[j] + replace_cost;
                curr[j + 1] = deletion.min(insertion).min(replacement);
                row_min = row_min.min(curr[j + 1]);
            }
            if row_min > max_distance {
                return None;
            }
            std::mem::swap(&mut prev, &mut curr);
        }
        let dist = prev[b_chars.len()];
        (dist <= max_distance).then_some(dist)
    }

    fn collapse_vowel_runs(input: &str) -> String {
        let mut out = String::with_capacity(input.len());
        let mut prev: Option<char> = None;
        for c in input.chars() {
            let is_vowel = matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U');
            if is_vowel && prev == Some(c) {
                continue;
            }
            out.push(c);
            prev = Some(c);
        }
        out
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_dictionary(&self) -> Result<(), std::io::Error> {
        if let Some(path) = &self.dictionary_path {
            save_to_disk(self, std::path::Path::new(path))
        } else {
            Ok(())
        }
    }
    #[cfg(target_arch = "wasm32")]
    pub fn save_dictionary(&self) -> Result<(), std::io::Error> {
        Ok(())
    }
}

impl Default for ImeEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn data_path(file: &str) -> PathBuf {
    if let Ok(dir) = std::env::var("AKSHAR_DATA_DIR") {
        let p = std::path::Path::new(&dir).join(file);
        if p.exists() {
            return p;
        }
    }
    let repo = std::path::Path::new("data").join(file);
    if repo.exists() {
        return repo;
    }
    if let Some(home) = dirs::home_dir() {
        let p = home.join(".local/share/akshar-ime").join(file);
        if p.exists() {
            return p;
        }
    }
    std::path::Path::new("/usr/share/akshar-ime").join(file)
}

#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
fn data_path(_file: &str) -> PathBuf {
    PathBuf::new()
}

#[cfg(not(target_arch = "wasm32"))]
fn load_model_or_default() -> TranslitModel {
    match TranslitModel::load(&data_path("translit_model.bin")) {
        Ok(m) if m.validate() => m,
        _ => TranslitModel::default(),
    }
}
#[cfg(target_arch = "wasm32")]
fn load_model_or_default() -> TranslitModel {
    TranslitModel::default()
}

#[cfg(not(target_arch = "wasm32"))]
fn load_lexicon() -> Option<RomanLexicon> {
    RomanLexicon::load(&data_path("roman_lexicon.bin")).ok()
}
#[cfg(target_arch = "wasm32")]
fn load_lexicon() -> Option<RomanLexicon> {
    None
}

fn load_reranker(lexicon: Option<RomanLexicon>) -> Reranker {
    let mut weights = [1.0f64; NUM_FEATURES];
    weights[0] = 1.0;
    weights[1] = 1.0;
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Ok(file) = std::fs::File::open(data_path("reranker_weights.json")) {
            if let Ok(v) = serde_json::from_reader::<_, serde_json::Value>(file) {
                if let Some(w) = v.get("weights") {
                    let names = crate::core::reranker::feature_names();
                    for (i, name) in names.iter().enumerate() {
                        if let Some(val) = w.get(*name).and_then(|x| x.as_f64()) {
                            weights[i] = val;
                        }
                    }
                }
            }
        }
    }
    let reranker = Reranker::new(weights, lexicon);
    // E3: word-frequency evidence (native targets only; absent on wasm unless
    // fetched separately).
    #[cfg(not(target_arch = "wasm32"))]
    let reranker = {
        let mut r = reranker;
        if let Ok(bytes) = std::fs::read(data_path("word_freq_text.bin")) {
            if let Ok(map) = bincode::deserialize::<HashMap<String, u32>>(&bytes) {
                r = r.with_freq(Some(map));
            }
        }
        r
    };
    reranker
}

/// Total corpus tokens across the vocabulary — the denominator of the unigram
/// P(w) in the PMI context term.  0.0 with no vocabulary, which disables the
/// term rather than dividing by zero.
fn total_tokens(data: Option<&crate::core::reranker::RerankerData>) -> f64 {
    data.map_or(0.0, |d| d.freq.values().map(|&f| f as f64).sum())
}

/// Reranker data: the corpus vocabulary plus its frequency-rank index.
/// None when the vocabulary file is unavailable (WASM-lite, fresh clones).
#[cfg(not(target_arch = "wasm32"))]
fn load_reranker_data() -> Option<crate::core::reranker::RerankerData> {
    let bytes = std::fs::read(data_path("word_freq_text.bin")).ok()?;
    crate::core::reranker::RerankerData::from_bin_bytes(&bytes)
}

#[cfg(target_arch = "wasm32")]
fn load_reranker_data() -> Option<crate::core::reranker::RerankerData> {
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn load_bigrams() -> Option<HashMap<String, Vec<(String, u32)>>> {
    let bytes = std::fs::read(data_path("word_bigrams.bin")).ok()?;
    bincode::deserialize(&bytes).ok()
}

#[cfg(target_arch = "wasm32")]
fn load_bigrams() -> Option<HashMap<String, Vec<(String, u32)>>> {
    None
}


#[cfg(test)]
mod tests {
    use super::*;

    // A decoder built from a tiny hand-built model, so tests don't depend on
    // the on-disk model file.
    fn tiny_decoder() -> ModelDecoder {
        let mut m = TranslitModel {
            version: crate::core::translit_model::MODEL_VERSION,
            ..TranslitModel::default()
        };
        // aksharas: 0=क 1=कि 2=न 3=म 4=मा 5=स्ते 6=ने 7=प 8=आल 9=र
        for a in ["क", "कि", "न", "म", "मा", "स्ते", "ने", "प", "आल", "र"]
        {
            m.aksharas.push(a.to_string());
        }
        for chunk in [
            "ka", "ki", "na", "ma", "maa", "ste", "ne", "pa", "aal", "ra",
        ] {
            m.chunks.push(chunk.to_string());
        }
        // emissions: chunk id -> akshara id
        let chunk = |s: &str| m.chunks.iter().position(|c| c == s).unwrap() as u32;
        let emit = |_a: u32, c: &str, w: f32| vec![(chunk(c), w)];
        m.emissions = vec![
            emit(0, "ka", 0.1),  // क -> ka
            emit(1, "ki", 0.1),  // कि -> ki
            emit(2, "na", 0.1),  // न -> na
            emit(3, "ma", 0.2),  // म -> ma
            emit(4, "maa", 0.1), // मा -> maa
            emit(5, "ste", 0.1), // स्ते -> ste
            emit(6, "ne", 0.1),  // ने -> ne
            emit(7, "pa", 0.1),  // प -> pa
            emit(8, "aal", 0.1), // आल -> aal
            emit(9, "ra", 0.1),  // र -> ra
        ];
        m.bigrams = vec![vec![]; 10];
        m.backoff = vec![6.0; 10];
        m.unigram_kn = vec![4.0; 10];
        m.word_start = vec![4.0; 10];
        m.build_trigram_index();
        ModelDecoder::with_config(m, DecoderConfig::default())
    }

    fn tiny_lexicon() -> RomanLexicon {
        RomanLexicon::build(vec![
            ("namaste".to_string(), "नमस्ते".to_string()),
            ("nepal".to_string(), "नेपाल".to_string()),
        ])
    }

    fn engine_with(model: TranslitModel, lexicon: Option<RomanLexicon>) -> ImeEngine {
        let decoder = ModelDecoder::with_config(
            model,
            DecoderConfig {
                beam_width: 64,
                ..DecoderConfig::default()
            },
        );
        ImeEngine {
            decoder,
            reranker: Reranker::default(),
            lexicon,
            reranker_data: None,
            word_trie: None,
            corpus_bigrams: None,
            bigram_context_enabled: true,
            bigram_weight: DEFAULT_BIGRAM_WEIGHT,
            bigram_shrink: DEFAULT_BIGRAM_SHRINK,
            vocab_total: 0.0,
            last_word: None,
            trie: Trie::new(),
            context_model: ContextModel::new(3),
            symspell: SymSpell::new(2),
            transliteration_model: HashMap::new(),
            learning_engine: LearningEngine::new(),
            dictionary_path: None,
            sparse_table: None,
        }
    }

    #[test]
    fn suggestions_include_decoder_fresh_transliteration() {
        let engine = engine_with(tiny_decoder().model, None);
        let suggestions = engine.get_suggestions("namaste", 8);
        assert!(suggestions.iter().any(|(d, _)| d == "नमस्ते"));
    }

    #[test]
    fn lexicon_exact_match_boosts_word_above_fresh() {
        // "namaste" isn't in the tiny decoder's vocab as a full word path, but
        // the lexicon has it; the exact-lexicon score should surface it.
        let engine = engine_with(tiny_decoder().model, Some(tiny_lexicon()));
        let suggestions = engine.get_suggestions("namaste", 8);
        assert!(suggestions.iter().any(|(d, _)| d == "नमस्ते"));
    }

    #[test]
    fn user_confirmation_moves_word_up() {
        let mut engine = engine_with(tiny_decoder().model, None);
        engine.user_confirms("namaste", "नमस्ते");
        engine.user_confirms("namaste", "नमस्ते");
        let suggestions = engine.get_suggestions("namaste", 8);
        let pos = suggestions
            .iter()
            .position(|(d, _)| d == "नमस्ते")
            .expect("नमस्ते should be suggested after learning");
        assert_eq!(pos, 0, "learned word should rank first");
    }

    #[test]
    fn empty_prefix_returns_nothing() {
        let engine = engine_with(tiny_decoder().model, None);
        assert!(engine.get_suggestions("", 8).is_empty());
    }

    #[test]
    fn bounded_levenshtein_respects_max_distance() {
        assert_eq!(ImeEngine::bounded_levenshtein("kal", "kal", 2), Some(0));
        assert_eq!(ImeEngine::bounded_levenshtein("kal", "kall", 2), Some(1));
        assert_eq!(ImeEngine::bounded_levenshtein("kal", "xyz", 2), None);
    }

    #[test]
    fn lone_dot_is_purnabiram() {
        let engine = engine_with(tiny_decoder().model, None);
        let out = engine.get_suggestions(".", 8);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "।");
    }

    #[test]
    fn trailing_dot_appends_purnabiram() {
        let engine = engine_with(tiny_decoder().model, None);
        let out = engine.get_suggestions("namaste.", 8);
        assert!(!out.is_empty(), "purnabiram input should still decode");
        assert!(out.iter().all(|(d, _)| d.ends_with('।')));
        // and the plain word still decodes identically underneath
        let plain: Vec<String> = engine
            .get_suggestions("namaste", 8)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        let dotted: Vec<String> = out
            .into_iter()
            .map(|(d, _)| d.strip_suffix('।').map(|s| s.to_string()).unwrap_or(d))
            .collect();
        assert_eq!(
            plain, dotted,
            "suggestions with '.' must equal plain suggestions + ।"
        );
    }

    #[test]
    fn all_digits_map_to_devanagari() {
        let engine = engine_with(tiny_decoder().model, None);
        let out = engine.get_suggestions("123", 8);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "१२३");
        // year with purnabiram
        let out = engine.get_suggestions("2081.", 8);
        assert_eq!(out[0].0, "२०८१।");
    }

    #[test]
    fn trailing_digits_map_onto_suggestions() {
        let engine = engine_with(tiny_decoder().model, None);
        let out = engine.get_suggestions("namaste1", 8);
        assert!(out.iter().any(|(d, _)| d.ends_with('१')));
        assert!(out.iter().any(|(d, _)| d.starts_with("नमस्ते")));
    }

    #[test]
    fn leading_digits_prepend_mapping() {
        let engine = engine_with(tiny_decoder().model, None);
        let out = engine.get_suggestions("12na", 8);
        assert!(!out.is_empty());
        assert!(out.iter().all(|(d, _)| d.starts_with("१२")));
    }

    #[test]
    fn digits_between_words_interleave() {
        let engine = engine_with(tiny_decoder().model, None);
        let out = engine.get_suggestions("na2ma", 8);
        assert!(!out.is_empty());
        assert_eq!(out[0].0, "न२म");
    }
}
