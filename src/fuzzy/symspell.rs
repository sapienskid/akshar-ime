// File: src/fuzzy/symspell.rs
use crate::core::types::WordId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A high-performance fuzzy search and spelling correction engine based on the
/// Symmetric Delete (SymSpell) algorithm. It pre-calculates a dictionary of "deletes"
/// for O(1) lookup complexity (relative to dictionary size).
#[derive(Clone, Serialize, Deserialize)]
pub struct SymSpell {
    /// Maps a delete variant (e.g., "namste") to the list of original WordIds
    /// it could have come from (e.g., [id_for_namaste]).
    deletes: HashMap<String, HashSet<WordId>>,
    max_edit_distance: usize,
}

impl SymSpell {
    pub fn new(max_edit_distance: usize) -> Self {
        Self {
            deletes: HashMap::new(),
            max_edit_distance,
        }
    }

    /// Adds a word to the SymSpell dictionary by generating all its delete variants
    /// up to the configured edit distance and mapping them back to the word's ID.
    /// Complexity: Amortized O(k^2) where k is the word length, due to string operations.
    pub fn add_word(&mut self, word: &str, word_id: WordId) {
        let edits = self.generate_edits(word);
        for edit in edits {
            self.deletes.entry(edit).or_default().insert(word_id);
        }
    }

    /// Looks up a potentially misspelled word by generating its deletes and
    /// finding them in the pre-calculated dictionary.
    /// Complexity: O(k^2) where k is the input length. Crucially, this is
    /// independent of the main dictionary size, making it extremely fast.
    pub fn lookup(&self, input: &str) -> HashSet<WordId> {
        let mut candidates = HashSet::new();
        
        // Check for an exact match first
        if let Some(word_ids) = self.deletes.get(input) {
            for &id in word_ids {
                candidates.insert(id);
            }
        }

        // Check for matches within the edit distance
        let edits = self.generate_edits(input);
        for edit in edits {
            if let Some(word_ids) = self.deletes.get(&edit) {
                for &id in word_ids {
                    candidates.insert(id);
                }
            }
        }
        candidates
    }

    /// Generates all unique string variants within the max_edit_distance.
    /// This includes the original string itself.
    fn generate_edits(&self, word: &str) -> HashSet<String> {
        let mut edits = HashSet::new();
        edits.insert(word.to_string()); // Distance 0

        let mut current_edits = edits.clone();

        for _ in 0..self.max_edit_distance {
            let mut next_edits = HashSet::new();
            for edit in current_edits {
                for i in 0..edit.len() {
                    let mut deleted_variant = edit.clone();
                    deleted_variant.remove(i);
                    next_edits.insert(deleted_variant);
                }
            }
            edits.extend(next_edits.clone());
            current_edits = next_edits;
        }
        
        edits
    }
}