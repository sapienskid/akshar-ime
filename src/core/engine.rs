use crate::core::{context::ContextModel, converter::RomanizationEngine, trie::TrieBuilder};
use crate::learning::{LearningEngine, WordConfirmation};
use crate::persistence::{load_from_disk, save_to_disk};
use std::path::Path;
use crate::core::types::WordId;
use std::collections::HashMap;
const CONTEXT_WINDOW_SIZE: usize = 3;

// The main IME engine is now composed of the builder and other models.
// NOTE: We no longer derive Serialize/Deserialize here. We have custom persistence logic.
pub struct ImeEngine {
    pub trie_builder: TrieBuilder,
    pub context_model: ContextModel,
    pub romanizer: RomanizationEngine,
    learning_engine: LearningEngine,
    dictionary_path: Option<String>,
}

impl ImeEngine {
    pub fn new() -> Self {
        Self {
            trie_builder: TrieBuilder::new(),
            context_model: ContextModel::new(CONTEXT_WINDOW_SIZE),
            romanizer: RomanizationEngine::new(),
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

        // 1. Get LEARNED suggestions from the Pruning Radix Trie
        let mut learned_suggestions: HashMap<String, u64> = self
            .trie_builder
            .get_top_k_suggestions(prefix, count * 2) // Get more to allow for ranking
            .into_iter()
            .map(|(id, score)| {
                let nepali_word = self.trie_builder.metadata_store[id].nepali.clone();
                (nepali_word, score)
            })
            .collect();

        // 2. Generate TRANSLITERATION candidates from the rules engine
        let translit_candidates = self.romanizer.generate_candidates(prefix);

        // 3. MERGE learned suggestions and transliterations
        for candidate in translit_candidates {
            // If a transliteration matches a learned word, use its score.
            // Otherwise, give it a base score (e.g., 1) to show it as a new option.
            learned_suggestions.entry(candidate).or_insert(1);
        }

        // 4. Convert to Vec for sorting and context re-ranking
        let mut all_suggestions: Vec<(String, u64)> = learned_suggestions.into_iter().collect();

        // 5. Create WordIds for context ranking (this is a bit inefficient but clear)
        let mut suggestions_with_ids: Vec<(WordId, u64)> = all_suggestions
            .iter()
            .filter_map(|(nepali, score)| {
                self.trie_builder
                    .find_word_id_by_nepali(nepali)
                    .map(|id| (id, *score))
            })
            .collect();

        // 6. Apply CONTEXT-BASED re-ranking to the learned words
        self.context_model
            .rerank_suggestions(&mut suggestions_with_ids);

        // 7. Update the scores in the main suggestion list
        for (id, new_score) in suggestions_with_ids {
            let nepali_word = &self.trie_builder.metadata_store[id].nepali;
            if let Some(entry) = all_suggestions.iter_mut().find(|(s, _)| s == nepali_word) {
                entry.1 = new_score;
            }
        }

        // 8. Sort the final merged list by score (descending)
        all_suggestions.sort_by_key(|&(_, score)| std::cmp::Reverse(score));

        // 9. Return the top K results
        all_suggestions.into_iter().take(count).collect()
    }

    pub fn user_confirms(&mut self, roman: &str, nepali: &str) {
        if roman.is_empty() || nepali.is_empty() {
            return;
        }
        let confirmation = WordConfirmation {
            roman: roman.to_string(),
            nepali: nepali.to_string(),
        };
        self.learning_engine.learn(
            &mut self.trie_builder,
            &mut self.context_model,
            &confirmation,
        );
    }

    pub fn save_dictionary(&self) -> Result<(), std::io::Error> {
        if let Some(path) = &self.dictionary_path {
            save_to_disk(self, Path::new(path))
        } else {
            Ok(()) // Don't error if no path is set
        }
    }
}
