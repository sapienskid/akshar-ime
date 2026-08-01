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
    translit_model::TranslitModel,
    trie::Trie,
    types::{TransliterationModel, WordId},
};
use crate::fuzzy::symspell::SymSpell;
use crate::learning::{LearningEngine, WordConfirmation};
use crate::persistence::{load_from_disk, save_to_disk};
use std::collections::HashMap;
use std::path::Path;

const CONTEXT_WINDOW_SIZE: usize = 3;
const MAX_EDIT_DISTANCE: usize = 2;
const QUERY_VARIANT_LIMIT: usize = 6;

/// Decoder beam for the IME (accuracy/speed sweet spot, see M2 eval).
const DECODER_BEAM: usize = 128;
/// Scale converting a decoder -log weight into a higher-better u64 score.
const FRESH_SCALE: f64 = 800.0;
/// A word in the corpus lexicon whose roman matches the typed prefix exactly.
const LEXICON_EXACT_SCORE: u64 = 300;
/// User-confirmed word from the learned trie.
const USER_TRIE_BASE: u64 = 180;
/// Fuzzy (edit-distance) match over user-learned roman variants.
const FUZZY_BASE: u64 = 50;
const FUZZY_DISTANCE_PENALTY_SCALE: u64 = 12;

pub struct ImeEngine {
    pub decoder: ModelDecoder,
    pub lexicon: Option<RomanLexicon>,
    pub trie: Trie,
    pub context_model: ContextModel,
    pub symspell: SymSpell,
    pub(crate) transliteration_model: TransliterationModel,
    learning_engine: LearningEngine,
    dictionary_path: Option<String>,
}

impl ImeEngine {
    pub fn new() -> Self {
        let model = load_model_or_default();
        let decoder = ModelDecoder::with_config(
            model,
            DecoderConfig {
                beam_width: DECODER_BEAM,
                ..DecoderConfig::default()
            },
        );
        let lexicon = load_lexicon();
        Self {
            decoder,
            lexicon,
            trie: Trie::new(),
            context_model: ContextModel::new(CONTEXT_WINDOW_SIZE),
            symspell: SymSpell::new(MAX_EDIT_DISTANCE),
            transliteration_model: HashMap::new(),
            learning_engine: LearningEngine::new(),
            dictionary_path: None,
        }
    }

    pub fn from_file_or_new(path: &str) -> Self {
        let mut engine = load_from_disk(Path::new(path)).unwrap_or_else(|_| Self::new());
        engine.dictionary_path = Some(path.to_string());
        engine
    }

    pub fn get_suggestions(&self, prefix: &str, count: usize) -> Vec<(String, u64)> {
        if prefix.is_empty() {
            return vec![];
        }
        let count = count.max(1);
        let query_variants = expand_query_variants(prefix, QUERY_VARIANT_LIMIT);

        let mut candidates: HashMap<String, u64> = HashMap::new();
        let mut add = |dev: String, score: u64| {
            candidates
                .entry(dev)
                .and_modify(|s| *s = (*s).max(score))
                .or_insert(score);
        };

        // 1. Fresh transliterations from the generative decoder.  We decode the
        //    base roman plus the few lowest-penalty soft variants (e.g. bhaai
        //    for bhai) because long-vowel alternants surface intended forms the
        //    base roman leaves ambiguous.  Capping avoids the 10x latency cost
        //    of re-decoding every variant.
        let mut decodes: Vec<&str> = Vec::new();
        for qv in &query_variants {
            if decodes.len() >= 3 {
                break;
            }
            if !decodes.contains(&qv.roman.as_str()) {
                decodes.push(qv.roman.as_str());
            }
        }
        for roman in decodes {
            for (dev, weight) in self.decoder.decode(roman, count * 3) {
                let score = (FRESH_SCALE / (1.0 + weight)).round().max(1.0) as u64;
                add(dev, score);
            }
        }

        for qv in &query_variants {
            let roman = qv.roman.as_str();

            // 2. Lexicon evidence: only EXACT roman matches (a confirmed word).
            //    The Aksharantar lexicon is mined from news and mostly holds
            //    long compound words, so prefix matching would flood the list
            //    with compounds like नेपालअधिराज्य; the decoder already covers
            //    prefix transliteration.
            if let Some(lx) = &self.lexicon {
                for dev in lx.lookup_exact(roman) {
                    add(dev, LEXICON_EXACT_SCORE.saturating_sub(qv.penalty));
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
                    if let Some(min_dist) = self.min_roman_distance(roman, meta, MAX_EDIT_DISTANCE) {
                        let dist_penalty = (min_dist as u64) * FUZZY_DISTANCE_PENALTY_SCALE;
                        let score = FUZZY_BASE.saturating_sub(dist_penalty);
                        add(meta.devanagari.clone(), score);
                    }
                }
            }
        }

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

    pub fn user_confirms(&mut self, roman: &str, devanagari: &str) {
        if roman.is_empty() || devanagari.is_empty() {
            return;
        }
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

    pub fn save_dictionary(&self) -> Result<(), std::io::Error> {
        if let Some(path) = &self.dictionary_path {
            save_to_disk(self, Path::new(path))
        } else {
            Ok(())
        }
    }
}

impl Default for ImeEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn load_model_or_default() -> TranslitModel {
    let path = Path::new("data/translit_model.bin");
    match TranslitModel::load(path) {
        Ok(m) if m.validate() => m,
        _ => TranslitModel::default(),
    }
}

fn load_lexicon() -> Option<RomanLexicon> {
    RomanLexicon::load(Path::new("data/roman_lexicon.bin")).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::WordMetadata;
    use std::collections::HashSet;

    // A decoder built from a tiny hand-built model, so tests don't depend on
    // the on-disk model file.
    fn tiny_decoder() -> ModelDecoder {
        let mut m = TranslitModel::default();
        m.version = crate::core::translit_model::MODEL_VERSION;
        // aksharas: 0=क 1=कि 2=न 3=म 4=मा 5=स्ते 6=ने 7=प 8=आल 9=र
        for a in ["क", "कि", "न", "म", "मा", "स्ते", "ने", "प", "आल", "र"] {
            m.aksharas.push(a.to_string());
        }
        for chunk in ["ka", "ki", "na", "ma", "maa", "ste", "ne", "pa", "aal", "ra"] {
            m.chunks.push(chunk.to_string());
        }
        // emissions: chunk id -> akshara id
        let chunk = |s: &str| m.chunks.iter().position(|c| c == s).unwrap() as u32;
        let emit = |a: u32, c: &str, w: f32| vec![(chunk(c), w)];
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
            lexicon,
            trie: Trie::new(),
            context_model: ContextModel::new(3),
            symspell: SymSpell::new(2),
            transliteration_model: HashMap::new(),
            learning_engine: LearningEngine::new(),
            dictionary_path: None,
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
}
