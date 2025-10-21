// File: src/core/engine.rs
use crate::core::{
    context::ContextModel, converter::RomanizationEngine,
    trie::Trie, types::WordId,
};
use crate::fuzzy::symspell::SymSpell;
use crate::learning::{LearningEngine, WordConfirmation};
use crate::persistence::{load_from_disk, save_to_disk};
use std::collections::HashMap;
use std::path::Path;

const CONTEXT_WINDOW_SIZE: usize = 3;
const MAX_EDIT_DISTANCE: usize = 2;

const LITERAL_BASE_SCORE: u64 = 1;
const PRIMARY_LITERAL_SCORE: u64 = 2;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
enum SuggestionSource {
    Literal,
    PrimaryLiteral,
    Fuzzy,
    Trie,
}

pub struct ImeEngine {
    pub trie: Trie,
    pub context_model: ContextModel,
    pub romanizer: RomanizationEngine,
    pub symspell: SymSpell,
    learning_engine: LearningEngine,
    dictionary_path: Option<String>,
}

impl ImeEngine {
    pub fn new() -> Self {
        Self {
            trie: Trie::new(),
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

        // Helper closure to manage candidate insertion logic.
        let add_candidate = |devanagari: String, score: u64, source: SuggestionSource, current_candidates: &mut HashMap<String, (u64, SuggestionSource)>| {
            current_candidates.entry(devanagari)
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

        // --- Stage 1: Trie Search ---
        let trie_suggestions = self.trie.get_top_k_suggestions(prefix, count);
        for (word_id, score) in trie_suggestions {
            if let Some(metadata) = self.trie.metadata_store.get(word_id) {
                add_candidate(metadata.devanagari.clone(), score, SuggestionSource::Trie, &mut candidates);
            }
        }

        // --- BORROW CHECKER FIX IS HERE ---
        // The original code had a long-lived closure that caused a borrow conflict.
        // This new structure separates the fuzzy search logic, ensuring borrows do not overlap.
        // --- Stage 2: Fuzzy Search ---
        if candidates.len() < count {
            let fuzzy_matches = self.symspell.lookup(prefix);
            for word_id in fuzzy_matches {
                if let Some(metadata) = self.trie.metadata_store.get(word_id) {
                    let score = metadata.frequency.saturating_sub(1);
                    add_candidate(metadata.devanagari.clone(), score, SuggestionSource::Fuzzy, &mut candidates);
                }
            }
        }
        
        // --- Stage 3: Primary Rule-Based Transliteration ---
    let primary_devanagari = self.romanizer.transliterate_primary(prefix);
    add_candidate(primary_devanagari, PRIMARY_LITERAL_SCORE, SuggestionSource::PrimaryLiteral, &mut candidates);

        // --- Stage 4: Other Literal FSM Candidates ---
        let literal_candidates = self.romanizer.generate_candidates(prefix);
        for devanagari in literal_candidates {
            add_candidate(devanagari, LITERAL_BASE_SCORE, SuggestionSource::Literal, &mut candidates);
        }

        // --- Stage 5: Conversion, Contextual Re-ranking, and Final Sort ---
        let mut all_suggestions: Vec<(String, u64)> = candidates
            .into_iter()
            .map(|(s, (score, _))| (s, score))
            .collect();

        let mut suggestions_with_ids: Vec<(WordId, u64)> = all_suggestions.iter()
            .filter_map(|(dev, score)| {
                self.trie.find_word_id_by_devanagari(dev).map(|id| (id, *score))
            })
            .collect();

        self.context_model.rerank_suggestions(&mut suggestions_with_ids);

        for (id, new_score) in suggestions_with_ids {
            if let Some(dev_word) = self.trie.metadata_store.get(id).map(|m| &m.devanagari) {
                 if let Some(entry) = all_suggestions.iter_mut().find(|(s, _)| s == dev_word) {
                    entry.1 = new_score;
                }
            }
        }

        all_suggestions.sort_by_key(|&(_, score)| std::cmp::Reverse(score));
        all_suggestions.truncate(count);
        all_suggestions
    }

    pub fn user_confirms(&mut self, roman: &str, devanagari: &str) {
        if roman.is_empty() || devanagari.is_empty() { return; }
        let confirmation = WordConfirmation { roman: roman.to_string(), devanagari: devanagari.to_string() };
        self.learning_engine.learn(&mut self.trie, &mut self.context_model, &mut self.symspell, &confirmation);
    }

    pub fn save_dictionary(&self) -> Result<(), std::io::Error> {
        if let Some(path) = &self.dictionary_path {
            save_to_disk(self, Path::new(path))
        } else {
            Ok(())
        }
    }
}

impl Default for ImeEngine { fn default() -> Self { Self::new() } }