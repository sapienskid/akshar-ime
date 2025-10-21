// File: src/learning.rs
use crate::core::{context::ContextModel, trie::Trie}; // MODIFIED
use crate::fuzzy::symspell::SymSpell;

pub struct LearningEngine {
    frequency_increment: u64,
}

pub struct WordConfirmation {
    pub roman: String,
    pub devanagari: String,
}

impl LearningEngine {
    pub fn new() -> Self {
        Self { frequency_increment: 1 }
    }

    pub fn learn(
        &self,
        trie: &mut Trie, // MODIFIED
        context_model: &mut ContextModel,
        symspell: &mut SymSpell,
        confirmation: &WordConfirmation,
    ) {
    let word_id = trie.get_or_create_metadata(&confirmation.devanagari);
        
        let metadata = &mut trie.metadata_store[word_id];
        metadata.frequency += self.frequency_increment;
        
        // Only add the variant if it's new, to avoid bloating the metadata store
        if metadata.variants.insert(confirmation.roman.clone()) {
            // OPTIMIZATION: Only add the primary Roman variant and the Devanagari word itself to the
            // fuzzy index. This keeps the SymSpell dictionary much smaller and faster than
            // indexing every single user-typed variant.
            symspell.add_word(&confirmation.roman, word_id);
            if metadata.variants.len() == 1 { // First time we see this word, add its Nepali form too
                 symspell.add_word(&confirmation.devanagari, word_id);
            }
        }
        
        let updated_freq = metadata.frequency;

        trie.insert(&confirmation.roman, word_id, updated_freq);

        context_model.add_word(word_id);
    }
}