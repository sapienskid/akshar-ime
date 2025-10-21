// File: src/core/engine.rs
use crate::core::{
    context::ContextModel, converter::RomanizationEngine,
    trie::TrieBuilder, types::WordId,
};
use crate::fuzzy::symspell::SymSpell;
use crate::learning::{LearningEngine, WordConfirmation};
use crate::persistence::{load_from_disk, save_to_disk};
use std::collections::HashMap;
use std::path::Path;

const CONTEXT_WINDOW_SIZE: usize = 3;
const MAX_EDIT_DISTANCE: usize = 2;

const LITERAL_BASE_SCORE: u64 = 1;
// Give the primary transliteration a slightly higher base score
// so it appears before other generated variants if no dictionary entry exists.
const PRIMARY_LITERAL_SCORE: u64 = 2;

/// Defines the origin of a suggestion to allow for intelligent ranking.
/// Higher variants are considered higher quality.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
enum SuggestionSource {
    Literal,        // A heuristic-based variant from the FST
    PrimaryLiteral, // The single, deterministic FST output
    Fuzzy,          // A match from SymSpell
    Trie,           // A direct prefix match from the learned dictionary (highest quality)
}

pub struct ImeEngine {
    pub trie_builder: TrieBuilder,
    pub context_model: ContextModel,
    pub romanizer: RomanizationEngine,
    pub symspell: SymSpell,
    learning_engine: LearningEngine,
    dictionary_path: Option<String>,
}

impl ImeEngine {
    pub fn new() -> Self {
        Self {
            trie_builder: TrieBuilder::new(),
            context_model: ContextModel::new(CONTEXT_WINDOW_SIZE),
            romanizer: RomanizationEngine::new(),
            symspell: SymSpell::new(MAX_EDIT_DISTANCE),
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
        if prefix.is_empty() { return vec![]; }

        let mut candidates: HashMap<String, (u64, SuggestionSource)> = HashMap::new();

        // Helper to add candidates while respecting suggestion source quality.
        // A higher-quality source (e.g., Trie) will always overwrite a lower one (e.g., Literal),
        // regardless of score.
        let mut add_candidate = |nepali: String, score: u64, source: SuggestionSource| {
            candidates.entry(nepali)
                .and_modify(|(existing_score, existing_source)| {
                    if source > *existing_source {
                        *existing_score = score;
                        *existing_source = source;
                    } else if source == *existing_source && score > *existing_score {
                        *existing_score = score;
                    }
                })
                .or_insert((score, source));
        };

        // --- Stage 1: Trie Search (Highest Quality) ---
        let trie_suggestions = self.trie_builder.get_top_k_suggestions(prefix, count);
        for (word_id, score) in trie_suggestions {
            if let Some(metadata) = self.trie_builder.metadata_store.get(word_id) {
                add_candidate(metadata.nepali.clone(), score, SuggestionSource::Trie);
            }
        }

        // --- Stage 2: Fuzzy Search ---
        let fuzzy_matches = self.symspell.lookup(prefix);
        for word_id in fuzzy_matches {
            if let Some(metadata) = self.trie_builder.metadata_store.get(word_id) {
                // Fuzzy matches are penalized slightly to rank below exact prefix matches.
                let score = metadata.frequency.saturating_sub(1);
                add_candidate(metadata.nepali.clone(), score, SuggestionSource::Fuzzy);
            }
        }

        // --- Stage 3: Primary Rule-Based Transliteration (Ground Truth) ---
        // This is the single, most direct transliteration from the FST. We add it with a
        // special priority to ensure it's always an option for the user.
        let primary_nepali = self.romanizer.transliterate_primary(prefix);
        add_candidate(primary_nepali, PRIMARY_LITERAL_SCORE, SuggestionSource::PrimaryLiteral);

        // --- Stage 4: Other Literal FSM Candidates (Fallback Heuristics) ---
        let literal_candidates = self.romanizer.generate_candidates(prefix);
        for nepali in literal_candidates {
            // This will only insert if the candidate isn't already present from a better source.
            add_candidate(nepali, LITERAL_BASE_SCORE, SuggestionSource::Literal);
        }

        // --- Stage 5: Conversion, Contextual Re-ranking, and Final Sort ---
        let mut all_suggestions: Vec<(String, u64)> = candidates
            .into_iter()
            .map(|(s, (score, _))| (s, score))
            .collect();

        // Contextual re-ranking
        let mut suggestions_with_ids: Vec<(WordId, u64)> = all_suggestions.iter()
            .filter_map(|(nepali, score)| {
                self.trie_builder.find_word_id_by_nepali(nepali).map(|id| (id, *score))
            })
            .collect();

        self.context_model.rerank_suggestions(&mut suggestions_with_ids);

        for (id, new_score) in suggestions_with_ids {
            let nepali_word = &self.trie_builder.metadata_store[id].nepali;
            if let Some(entry) = all_suggestions.iter_mut().find(|(s, _)| s == nepali_word) {
                entry.1 = new_score;
            }
        }

        all_suggestions.sort_by_key(|&(_, score)| std::cmp::Reverse(score));
        all_suggestions.truncate(count);
        all_suggestions
    }

    pub fn user_confirms(&mut self, roman: &str, nepali: &str) {
        if roman.is_empty() || nepali.is_empty() { return; }
        let confirmation = WordConfirmation { roman: roman.to_string(), nepali: nepali.to_string() };
        self.learning_engine.learn(&mut self.trie_builder, &mut self.context_model, &mut self.symspell, &confirmation);
    }

    pub fn save_dictionary(&self) -> Result<(), std::io::Error> {
        if let Some(path) = &self.dictionary_path {
            save_to_disk(self, Path::new(path))
        } else {
            // In a real scenario, log a warning here.
            Ok(())
        }
    }

}

impl Default for ImeEngine { fn default() -> Self { Self::new() } }