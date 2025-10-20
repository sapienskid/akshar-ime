// File: src/learning.rs
use crate::core::{context::ContextModel, trie::TrieBuilder};
use crate::fuzzy::symspell::SymSpell;

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

    pub fn learn(
        &self,
        trie_builder: &mut TrieBuilder,
        context_model: &mut ContextModel,
        symspell: &mut SymSpell,
        confirmation: &WordConfirmation,
    ) {
        let word_id = trie_builder.get_or_create_metadata(&confirmation.nepali);
        
        let metadata = &mut trie_builder.metadata_store[word_id];
        metadata.frequency += self.frequency_increment;
        metadata.variants.insert(confirmation.roman.clone());
        
        let updated_freq = metadata.frequency;

        trie_builder.insert(&confirmation.roman, word_id, updated_freq);

        symspell.add_word(&confirmation.roman, word_id);
        symspell.add_word(&confirmation.nepali, word_id);

        context_model.add_word(word_id);
    }
}