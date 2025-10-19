// --- File: src/core/trie.rs
use std::collections::HashSet;
use crate::core::types::{WordId, WordMetadata};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BinaryHeap};

// --- TrieBuilder: Used for real-time, in-memory learning ---

#[derive(Clone, Serialize, Deserialize)]
struct BuilderNode {
    children: HashMap<u8, usize>,
    word_id: Option<WordId>,
    max_freq_in_subtree: u64,
}

impl BuilderNode {
    fn new() -> Self {
        Self { children: HashMap::new(), word_id: None, max_freq_in_subtree: 0 }
    }
}

/// A mutable, in-memory trie optimized for fast insertions and updates.
/// This is the structure the engine interacts with during a user session.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrieBuilder {
    nodes: Vec<BuilderNode>,
    pub metadata_store: Vec<WordMetadata>,
}

impl TrieBuilder {
    pub fn new() -> Self {
        Self { nodes: vec![BuilderNode::new()], metadata_store: Vec::new() }
    }

    /// Finds a WordId by its canonical Nepali string.
    pub fn find_word_id_by_nepali(&self, nepali: &str) -> Option<WordId> {
        self.metadata_store.iter().position(|meta| meta.nepali == nepali)
    }

    /// Gets or creates metadata for a nepali word, returning its ID.
    pub fn get_or_create_metadata(&mut self, nepali: &str) -> WordId {
        if let Some(id) = self.find_word_id_by_nepali(nepali) {
            id
        } else {
            let new_meta = WordMetadata {
                nepali: nepali.to_string(),
                frequency: 0,
                variants: HashSet::new(),
            };
            self.metadata_store.push(new_meta);
            self.metadata_store.len() - 1
        }
    }
    
    /// Inserts a roman variant mapping to a specific WordId.
    /// O(k) complexity where k is key length.
    pub fn insert(&mut self, key: &str, word_id: WordId, _frequency: u64) {
    let mut node_idx = 0;
    let mut path = vec![0];
    for &byte in key.as_bytes() {
        // --- BORROW CHECKER FIX IS HERE ---
        let next_idx = if let Some(&id) = self.nodes[node_idx].children.get(&byte) {
            id
        } else {
            let new_node_id = self.nodes.len();
            self.nodes.push(BuilderNode::new());
            self.nodes[node_idx].children.insert(byte, new_node_id);
            new_node_id
        };
        // --- END OF FIX ---
        
        node_idx = next_idx;
        path.push(node_idx);
    }
    self.nodes[node_idx].word_id = Some(word_id);

    // Propagate max frequency up the path
    for &idx in path.iter().rev() {
        let mut max_freq = if let Some(id) = self.nodes[idx].word_id {
            self.metadata_store[id].frequency
        } else {
            0
        };
        
        // This part needs a fix too, to avoid borrowing issues.
        // We collect child frequencies first, then update.
        let child_max_freqs: Vec<u64> = self.nodes[idx].children.values()
            .map(|&child_idx| self.nodes[child_idx].max_freq_in_subtree)
            .collect();

        for freq in child_max_freqs {
            max_freq = max_freq.max(freq);
        }
        
        if max_freq > self.nodes[idx].max_freq_in_subtree {
             self.nodes[idx].max_freq_in_subtree = max_freq;
        } else {
            break;
        }
    }
}

    /// Get top K suggestions using pruning.
    /// O(k + S log K) where S is number of nodes visited.
    pub fn get_top_k_suggestions(&self, prefix: &str, k: usize) -> Vec<(WordId, u64)> {
        let mut node_idx = 0;
        for &byte in prefix.as_bytes() {
            if let Some(&next_idx) = self.nodes[node_idx].children.get(&byte) {
                node_idx = next_idx;
            } else {
                return vec![];
            }
        }

        let mut heap = BinaryHeap::new();
        self.dfs_search(node_idx, k, &mut heap);
        
        // Min-heap gives smallest first, so we reverse for descending frequency order
        heap.into_sorted_vec().into_iter().map(|(freq, id)| (id, freq)).rev().collect()
    }

    fn dfs_search(&self, node_idx: usize, k: usize, heap: &mut BinaryHeap<(u64, WordId)>) {
        let node = &self.nodes[node_idx];
        if let Some(id) = node.word_id {
            let freq = self.metadata_store[id].frequency;
            if heap.len() < k {
                heap.push((freq, id));
            } else if freq > heap.peek().unwrap().0 {
                heap.pop();
                heap.push((freq, id));
            }
        }
        
        let min_freq_in_heap = if heap.len() == k { heap.peek().unwrap().0 } else { 0 };

        for &child_idx in node.children.values() {
            if self.nodes[child_idx].max_freq_in_subtree > min_freq_in_heap {
                self.dfs_search(child_idx, k, heap);
            }
        }
    }
}