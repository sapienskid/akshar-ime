use crate::core::{
    context::ContextModel,
    converter::RomanizationEngine,
    trie::TrieBuilder,
    types::{WordId},
};
use crate::learning::{LearningEngine, WordConfirmation};
use crate::persistence::{load_from_disk, save_to_disk};
use std::path::Path;

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
        let mut suggestions = self.trie_builder.get_top_k_suggestions(prefix, count);
        
        // Apply context-based re-ranking
        self.context_model.rerank_suggestions(&mut suggestions);

        // Convert WordIds back to Nepali strings for display
        suggestions.into_iter().map(|(id, score)| {
            let nepali_word = self.trie_builder.metadata_store[id].nepali.clone();
            (nepali_word, score)
        }).collect()
    }

    pub fn user_confirms(&mut self, roman: &str, nepali: &str) {
        if roman.is_empty() || nepali.is_empty() { return; }
        let confirmation = WordConfirmation {
            roman: roman.to_string(),
            nepali: nepali.to_string(),
        };
        self.learning_engine.learn(&mut self.trie_builder, &mut self.context_model, &confirmation);
    }

    pub fn save_dictionary(&self) -> Result<(), std::io::Error> {
        if let Some(path) = &self.dictionary_path {
            save_to_disk(self, Path::new(path))
        } else {
            Ok(()) // Don't error if no path is set
        }
    }
}