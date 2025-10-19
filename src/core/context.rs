// File: src/core/context.rs
use crate::core::types::WordId;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextModel {
    window_size: usize,
    history: VecDeque<WordId>,
    /// Maps (prev_word_id, current_word_id) -> frequency
    bigrams: HashMap<(WordId, WordId), u64>,
}

impl ContextModel {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            history: VecDeque::with_capacity(window_size),
            bigrams: HashMap::new(),
        }
    }

    /// Adds a confirmed word to the context history and updates bigram counts.
    /// O(1) amortized complexity.
    pub fn add_word(&mut self, word_id: WordId) {
        if let Some(&prev_word_id) = self.history.back() {
            *self.bigrams.entry((prev_word_id, word_id)).or_insert(0) += 1;
        }

        if self.history.len() == self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(word_id);
    }

    /// Re-ranks a list of suggestions based on the current context.
    /// Suggestions that form common bigrams with the previous word get a score boost.
    pub fn rerank_suggestions(&self, suggestions: &mut Vec<(WordId, u64)>) {
        if let Some(&prev_word_id) = self.history.back() {
            for (word_id, score) in suggestions.iter_mut() {
                if let Some(&bigram_count) = self.bigrams.get(&(prev_word_id, *word_id)) {
                    // Simple boost: add a factor of the bigram count.
                    // A more advanced model might use logarithms or smoothed probabilities.
                    let boost = (bigram_count as f64).log2() * 10.0;
                    *score += boost as u64;
                }
            }
            // Re-sort the suggestions based on the new boosted scores
            suggestions.sort_by_key(|&(_, score)| std::cmp::Reverse(score));
        }
    }
}