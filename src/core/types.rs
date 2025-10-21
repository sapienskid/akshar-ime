// src/core/types.rs
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A unique identifier for a canonical Nepali word.
pub type WordId = usize;

/// Rich metadata associated with a single canonical Nepali word.
/// This is the "value" in our learned dictionary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordMetadata {
    pub nepali: String,
    /// Represents P(N) - the Language Model probability.
    /// This is the total number of times this word has been confirmed.
    pub frequency: u64,
    /// All Romanized spellings the user has used for this word.
    /// e.g., {"cha", "chha", "xa"} for "छ".
    pub variants: HashSet<String>,
}

/// A new model to store transliteration probabilities, P(R|N).
/// Maps a (Roman String, Nepali WordId) pair to its co-occurrence frequency.
pub type TransliterationModel = std::collections::HashMap<(String, WordId), u64>;