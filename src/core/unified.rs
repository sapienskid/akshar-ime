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
use std::io::BufWriter;
use std::path::Path;

pub const UNIFIED_MAGIC: [u8; 4] = *b"AKSH";
/// v2 adds `sparse_scale`.  v1 containers still load: they get the legacy
/// compile-time constant, which is what they were quantized against.
pub const UNIFIED_VERSION: u32 = 2;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UnifiedModel {
    pub magic: [u8; 4],
    pub version: u32,
    pub translit: TranslitModel,
    pub sparse_reranker_table: Vec<i8>,
    /// Dequantization scale for `sparse_reranker_table`.
    ///
    /// The trainer quantizes the f32 sparse weights with `127 / max|w|`, a
    /// value that depends on the run.  It used to compute that scale and throw
    /// it away, while inference multiplied by a hardcoded
    /// `reranker_weights::SPARSE_SCALE` emitted by `train_reranker.rs` -- a
    /// binary deleted in c0d12ba.  Every model trained since then has had its
    /// sparse contribution scaled by an unrelated constant.  Carrying the
    /// scale in the container is what makes the table mean anything.
    pub sparse_scale: f64,
    pub vocab_freq: HashMap<String, u32>,
    pub bigrams: Option<HashMap<String, Vec<(String, u32)>>>,
}

/// The v1 container layout, kept only so existing `akshar.model` files still
/// load.  bincode is not self-describing, so a new field cannot simply be
/// added with `#[serde(default)]` -- the old byte stream has to be parsed with
/// the exact struct it was written from.
#[derive(Deserialize)]
struct UnifiedModelV1 {
    magic: [u8; 4],
    /// Read by serde to consume the field; the converted struct always
    /// carries UNIFIED_VERSION instead.
    #[allow(dead_code)]
    version: u32,
    translit: TranslitModel,
    sparse_reranker_table: Vec<i8>,
    vocab_freq: HashMap<String, u32>,
    bigrams: Option<HashMap<String, Vec<(String, u32)>>>,
}

impl From<UnifiedModelV1> for UnifiedModel {
    fn from(v1: UnifiedModelV1) -> Self {
        Self {
            magic: v1.magic,
            // Once converted, the in-memory struct IS the v2 layout, so it must
            // say so: `save` serializes whatever is in `version`, and stamping
            // a v2 byte layout as v1 makes the file unreadable.
            version: UNIFIED_VERSION,
            translit: v1.translit,
            sparse_reranker_table: v1.sparse_reranker_table,
            // v1 tables were quantized against whatever scale the run
            // produced, but only the compile-time constant was ever applied
            // at inference.  Reproduce that exactly rather than silently
            // changing the behaviour of an already-shipped model.
            sparse_scale: crate::core::reranker_weights::SPARSE_SCALE,
            vocab_freq: v1.vocab_freq,
            bigrams: v1.bigrams,
        }
    }
}

/// Read the container version without deserializing the body.  bincode lays
/// fields out in declaration order with no padding, so the version is the
/// u32 immediately after the 4 magic bytes.
fn peek_version(bytes: &[u8]) -> Option<u32> {
    if bytes.len() < 8 || bytes[..4] != UNIFIED_MAGIC {
        return None;
    }
    Some(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]))
}

impl UnifiedModel {
    pub fn new(
        translit: TranslitModel,
        sparse_reranker_table: Vec<i8>,
        sparse_scale: f64,
        vocab_freq: HashMap<String, u32>,
        bigrams: Option<HashMap<String, Vec<(String, u32)>>>,
    ) -> Self {
        Self {
            magic: UNIFIED_MAGIC,
            version: UNIFIED_VERSION,
            translit,
            sparse_reranker_table,
            sparse_scale,
            vocab_freq,
            bigrams,
        }
    }

    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        // The whole container is one bincode blob, so a streaming read buys
        // nothing over reading the file and dispatching on its version.
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let version = peek_version(bytes)
            .ok_or("Invalid magic header: not an Akshar unified model")?;
        let mut model: Self = match version {
            2 => bincode::deserialize(bytes)?,
            1 => bincode::deserialize::<UnifiedModelV1>(bytes)?.into(),
            other => return Err(format!("Unsupported model version: {other}").into()),
        };
        if model.magic != UNIFIED_MAGIC {
            return Err("Invalid magic header: not an Akshar unified model".into());
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
            && (1..=UNIFIED_VERSION).contains(&self.version)
            && self.translit.validate()
            && !self.sparse_reranker_table.is_empty()
            && !self.vocab_freq.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model() -> UnifiedModel {
        let mut translit = TranslitModel {
            version: 1,
            aksharas: vec!["क".to_string()],
            chunks: vec!["ka".to_string()],
            emissions: vec![vec![(0u32, 0.5f32)]],
            bigrams: vec![vec![]],
            backoff: vec![0.0],
            unigram_kn: vec![0.0],
            word_start: vec![0.0],
            ..Default::default()
        };
        translit.build_trigram_index();
        let mut vocab = HashMap::new();
        vocab.insert("क".to_string(), 7u32);
        UnifiedModel::new(translit, vec![1i8, -2, 3], 0.25, vocab, None)
    }

    /// v2 bytes (as written before the compact encoding landed) must still read.
    #[test]
    fn v2_files_still_load() {
        let m = tiny_model();
        let v2_bytes = bincode::serialize(&UnifiedModelV2Ser {
            magic: UNIFIED_MAGIC,
            version: 2,
            translit: m.translit.clone(),
            sparse_reranker_table: m.sparse_reranker_table.clone(),
            sparse_scale: 0.25,
            vocab_freq: m.vocab_freq.clone(),
            bigrams: None,
        })
        .expect("serialize v2");
        let back = UnifiedModel::from_bytes(&v2_bytes).expect("read v2");
        assert_eq!(back.version, UNIFIED_VERSION);
        assert_eq!(back.sparse_scale, 0.25);
        assert_eq!(back.translit.aksharas, vec!["क".to_string()]);
    }

    #[test]
    fn v2_round_trips_through_bytes() {
        let m = tiny_model();
        let bytes = m.to_bytes().expect("serialize");
        let back = UnifiedModel::from_bytes(&bytes).expect("deserialize");
        assert_eq!(back.version, UNIFIED_VERSION);
        assert_eq!(back.sparse_scale, 0.25);
        assert_eq!(back.sparse_reranker_table, vec![1i8, -2, 3]);
        assert_eq!(back.vocab_freq.get("क"), Some(&7));
        assert!(back.validate());
    }

    /// A model read from v1 and written back must come back as valid v2.
    /// Regression: the conversion used to preserve `version: 1` while writing
    /// the v2 field layout, producing a file that could not be read at all.
    #[test]
    fn v1_upgrades_and_then_round_trips() {
        let m = tiny_model();
        let v1_bytes = bincode::serialize(&UnifiedModelV1Ser {
            magic: UNIFIED_MAGIC,
            version: 1,
            translit: m.translit.clone(),
            sparse_reranker_table: m.sparse_reranker_table.clone(),
            vocab_freq: m.vocab_freq.clone(),
            bigrams: None,
        })
        .expect("serialize v1");

        let upgraded = UnifiedModel::from_bytes(&v1_bytes).expect("read v1");
        assert_eq!(upgraded.version, UNIFIED_VERSION);
        assert_eq!(
            upgraded.sparse_scale,
            crate::core::reranker_weights::SPARSE_SCALE
        );

        let rewritten = upgraded.to_bytes().expect("serialize v2");
        let back = UnifiedModel::from_bytes(&rewritten).expect("re-read");
        assert_eq!(back.version, UNIFIED_VERSION);
        assert_eq!(back.vocab_freq.get("क"), Some(&7));
    }

    #[test]
    fn rejects_foreign_and_future_files() {
        assert!(UnifiedModel::from_bytes(b"NOPE\x01\x00\x00\x00").is_err());
        let mut future = UNIFIED_MAGIC.to_vec();
        future.extend_from_slice(&99u32.to_le_bytes());
        assert!(UnifiedModel::from_bytes(&future).is_err());
    }

    /// Writer-side mirror of UnifiedModelV2, so the test can produce genuine
    /// v2 bytes (the reader struct is Deserialize-only).
    #[derive(Serialize)]
    struct UnifiedModelV2Ser {
        magic: [u8; 4],
        version: u32,
        translit: TranslitModel,
        sparse_reranker_table: Vec<i8>,
        sparse_scale: f64,
        vocab_freq: HashMap<String, u32>,
        bigrams: Option<HashMap<String, Vec<(String, u32)>>>,
    }

    /// Writer-side mirror of UnifiedModelV1, so the test can produce genuine
    /// v1 bytes (the reader struct is Deserialize-only).
    #[derive(Serialize)]
    struct UnifiedModelV1Ser {
        magic: [u8; 4],
        version: u32,
        translit: TranslitModel,
        sparse_reranker_table: Vec<i8>,
        vocab_freq: HashMap<String, u32>,
        bigrams: Option<HashMap<String, Vec<(String, u32)>>>,
    }
}
