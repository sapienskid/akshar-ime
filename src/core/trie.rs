// File: src/core/trie.rs
use crate::core::types::{WordId, WordMetadata};
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Clone, Serialize, Deserialize)]
struct Node {
    children: HashMap<u8, usize>,
    word_id: Option<WordId>,
    max_freq_in_subtree: u64,
}

impl Node {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            word_id: None,
            max_freq_in_subtree: 0,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Trie {
    nodes: Vec<Node>,
    pub metadata_store: Vec<WordMetadata>,
}

impl Trie {
    pub fn new() -> Self {
        Self {
            nodes: vec![Node::new()],
            metadata_store: Vec::new(),
        }
    }

    pub fn find_word_id_by_devanagari(&self, devanagari: &str) -> Option<WordId> {
        self.metadata_store.iter().position(|meta| meta.devanagari == devanagari)
    }

    pub fn get_or_create_metadata(&mut self, devanagari: &str) -> WordId {
        if let Some(id) = self.find_word_id_by_devanagari(devanagari) {
            id
        } else {
            let new_meta = WordMetadata {
                devanagari: devanagari.to_string(),
                frequency: 0,
                variants: HashSet::new(),
            };
            self.metadata_store.push(new_meta);
            self.metadata_store.len() - 1
        }
    }

    // Corrected the unused variable warning
    pub fn insert(&mut self, key: &str, word_id: WordId, _frequency: u64) {
        let mut node_idx = 0;
        let mut path = vec![0];

        for &byte in key.as_bytes() {
            // --- BORROW CHECKER FIX IS HERE ---
            // The original code held a mutable borrow on a node while trying to modify the
            // parent `nodes` vector, which is not allowed. This new structure performs
            // the check and the modification in separate steps, respecting the borrow checker.
            let next_idx = if let Some(&child_idx) = self.nodes[node_idx].children.get(&byte) {
                child_idx
            } else {
                let new_node_idx = self.nodes.len();
                self.nodes.push(Node::new());
                self.nodes[node_idx].children.insert(byte, new_node_idx);
                new_node_idx
            };
            // --- END OF FIX ---
            node_idx = next_idx;
            path.push(node_idx);
        }
        self.nodes[node_idx].word_id = Some(word_id);

        for &idx in path.iter().rev() {
            let current_node_freq = self.nodes[idx]
                .word_id
                .map_or(0, |id| self.metadata_store[id].frequency);

            let max_child_freq = self.nodes[idx]
                .children
                .values()
                .map(|&child_idx| self.nodes[child_idx].max_freq_in_subtree)
                .max()
                .unwrap_or(0);
            
            let new_max_freq = current_node_freq.max(max_child_freq);

            if new_max_freq == self.nodes[idx].max_freq_in_subtree {
                break;
            }
            self.nodes[idx].max_freq_in_subtree = new_max_freq;
        }
    }

    pub fn get_top_k_suggestions(&self, prefix: &str, k: usize) -> Vec<(WordId, u64)> {
        let mut node_idx = 0;
        for &byte in prefix.as_bytes() {
            if let Some(&next_idx) = self.nodes[node_idx].children.get(&byte) {
                node_idx = next_idx;
            } else {
                return vec![];
            }
        }

        let mut heap = BinaryHeap::with_capacity(k + 1);
        self.dfs_pruning_search(node_idx, k, &mut heap);

        heap.into_iter().map(|(freq, id)| (id, freq)).collect()
    }

    fn dfs_pruning_search(&self, node_idx: usize, k: usize, heap: &mut BinaryHeap<(u64, WordId)>) {
        let node = &self.nodes[node_idx];

        if let Some(id) = node.word_id {
            let freq = self.metadata_store[id].frequency;
            if freq > 0 {
                if heap.len() < k {
                    heap.push((freq, id));
                } else if freq > heap.peek().unwrap().0 {
                    heap.pop();
                    heap.push((freq, id));
                }
            }
        }

        let min_freq_in_heap = if heap.len() == k { heap.peek().unwrap().0 } else { 0 };

        for &child_idx in node.children.values() {
            if self.nodes[child_idx].max_freq_in_subtree > min_freq_in_heap {
                self.dfs_pruning_search(child_idx, k, heap);
            }
        }
    }
}