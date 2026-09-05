// File: src/core/unified.rs
//
// Unified model container for Akshar Devanagari IME.
// Bundles:
//   1. Transliteration Model (EM emissions + Kneser-Ney syllable LM)
//   2. Sparse Reranker Table (2^20 quantized feature weights)
//   3. Vocabulary Frequency Map (unigram frequencies for reranking & trie)
//   4. Word Bigrams (context-aware next-word suggestions)
//
// All packaged into a single binary artifact (`akshar.model`).

use crate::core::translit_model::TranslitModel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

pub const UNIFIED_MAGIC: [u8; 4] = *b"AKSH";
pub const UNIFIED_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UnifiedModel {
    pub magic: [u8; 4],
    pub version: u32,
    pub translit: TranslitModel,
    pub sparse_reranker_table: Vec<i8>,
    pub vocab_freq: HashMap<String, u32>,
    pub bigrams: Option<HashMap<String, Vec<(String, u32)>>>,
}

impl UnifiedModel {
    pub fn new(
        translit: TranslitModel,
        sparse_reranker_table: Vec<i8>,
        vocab_freq: HashMap<String, u32>,
        bigrams: Option<HashMap<String, Vec<(String, u32)>>>,
    ) -> Self {
        Self {
            magic: UNIFIED_MAGIC,
            version: UNIFIED_VERSION,
            translit,
            sparse_reranker_table,
            vocab_freq,
            bigrams,
        }
    }

    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let f = File::open(path)?;
        let reader = BufReader::new(f);
        let mut model: Self = bincode::deserialize_from(reader)?;
        if model.magic != UNIFIED_MAGIC {
            return Err("Invalid magic header: not an Akshar unified model".into());
        }
        if model.version != UNIFIED_VERSION {
            return Err(format!("Unsupported model version: {}", model.version).into());
        }
        model.translit.build_trigram_index();
        Ok(model)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let mut model: Self = bincode::deserialize(bytes)?;
        if model.magic != UNIFIED_MAGIC {
            return Err("Invalid magic header: not an Akshar unified model".into());
        }
        if model.version != UNIFIED_VERSION {
            return Err(format!("Unsupported model version: {}", model.version).into());
        }
        model.translit.build_trigram_index();
        Ok(model)
    }

    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let f = File::create(path)?;
        let writer = BufWriter::new(f);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Ok(bincode::serialize(self)?)
    }

    pub fn validate(&self) -> bool {
        self.magic == UNIFIED_MAGIC
            && self.version == UNIFIED_VERSION
            && self.translit.validate()
            && !self.sparse_reranker_table.is_empty()
            && !self.vocab_freq.is_empty()
    }
}
