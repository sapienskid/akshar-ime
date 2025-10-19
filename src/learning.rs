// --- File: src/learning.rs
use crate::core::{
    context::ContextModel,
    trie::TrieBuilder,
};

pub struct LearningEngine {
    frequency_increment: u64,
}

pub struct WordConfirmation {
    pub roman: String,
    pub nepali: String,
}

impl LearningEngine {
    pub fn new() -> Self {
        Self { frequency_increment: 1 }
    }

    /// The main learning function, now much more powerful.
    /// It updates frequencies, learns new romanization variants, and updates the context model.
    pub fn learn(
        &self,
        trie_builder: &mut TrieBuilder,
        context_model: &mut ContextModel,
        confirmation: &WordConfirmation,
    ) {
        // 1. Find or create the canonical Nepali word's metadata
        let word_id = trie_builder.get_or_create_metadata(&confirmation.nepali);
        
        // 2. Update metadata
        let metadata = &mut trie_builder.metadata_store[word_id];
        metadata.frequency += self.frequency_increment;
        metadata.variants.insert(confirmation.roman.clone());
        
        let updated_freq = metadata.frequency;

        // 3. Update the trie with the roman variant -> word_id mapping.
        // This will create the entry if it doesn't exist, or update the
        // max_freq path if it does.
        trie_builder.insert(&confirmation.roman, word_id, updated_freq);

        // 4. Update the context model with the confirmed word_id
        context_model.add_word(word_id);
    }
}