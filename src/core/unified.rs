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

use crate::core::codec;
use crate::core::translit_model::TranslitModel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

pub const UNIFIED_MAGIC: [u8; 4] = *b"AKSH";
/// Container version.
///
///   v1  original bincode blob
///   v2  adds `sparse_scale` (the trainer's measured dequantization scale)
///   v3  n-gram tables stored compactly: CSR + delta varints + an 8-bit
///       codebook per table, via `core::codec`
///
/// All three still load.  v3 is what `save` writes.
pub const UNIFIED_VERSION: u32 = 3;

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
    /// `reranker_weights::SPARSE_SCALE` emitted by `train_reranker.rs` — a
    /// binary deleted in c0d12ba.  Every model trained since then has had its
    /// sparse contribution scaled by an unrelated constant.  Carrying the
    /// scale in the container is what makes the table mean anything.
    #[serde(default)]
    pub sparse_scale: f64,
    pub vocab_freq: HashMap<String, u32>,
    pub bigrams: Option<HashMap<String, Vec<(String, u32)>>>,
}

/// The v1 container layout, kept only so existing `akshar.model` files still
/// load.  bincode is not self-describing, so a new field cannot simply be
/// added with `#[serde(default)]` — the old byte stream has to be parsed with
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

/// The v2 container layout: the v1 fields plus `sparse_scale`.
///
/// Spelled out rather than reusing `UnifiedModel` even though the two
/// currently match field for field.  `save` now writes v3, so nothing would
/// catch it if `UnifiedModel` drifted and silently broke v2 reads — which is
/// precisely the failure this file already had once.
#[derive(Deserialize)]
struct UnifiedModelV2 {
    magic: [u8; 4],
    #[allow(dead_code)]
    version: u32,
    translit: TranslitModel,
    sparse_reranker_table: Vec<i8>,
    sparse_scale: f64,
    vocab_freq: HashMap<String, u32>,
    bigrams: Option<HashMap<String, Vec<(String, u32)>>>,
}

impl From<UnifiedModelV2> for UnifiedModel {
    fn from(v2: UnifiedModelV2) -> Self {
        Self {
            magic: v2.magic,
            version: UNIFIED_VERSION,
            translit: v2.translit,
            sparse_reranker_table: v2.sparse_reranker_table,
            sparse_scale: v2.sparse_scale,
            vocab_freq: v2.vocab_freq,
            bigrams: v2.bigrams,
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
            3 => bincode::deserialize::<UnifiedModelV3>(bytes)?.try_into()?,
            2 => bincode::deserialize::<UnifiedModelV2>(bytes)?.into(),
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
        let mut writer = BufWriter::new(f);
        bincode::serialize_into(&mut writer, &UnifiedModelV3::from(self))?;
        Ok(())
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Ok(bincode::serialize(&UnifiedModelV3::from(self))?)
    }

    pub fn validate(&self) -> bool {
        self.magic == UNIFIED_MAGIC
            && (1..=UNIFIED_VERSION).contains(&self.version)
            && self.translit.validate()
            && !self.sparse_reranker_table.is_empty()
            && !self.vocab_freq.is_empty()
    }
}

// ---------------------------------------------------------------------------
// v3: compact n-gram encoding
// ---------------------------------------------------------------------------

/// The v3 body.
///
/// Everything that is *large and numeric* moves into opaque byte blobs
/// produced by `core::codec`; everything small and structural stays as
/// ordinary bincode fields.  The split is deliberate — the vocabulary strings
/// and the akshara list compress well under Brotli already, whereas the
/// n-gram weights do not (f32 mantissas are noise), so those are the ones
/// worth re-encoding by hand.
#[derive(Serialize, Deserialize)]
struct UnifiedModelV3 {
    magic: [u8; 4],
    version: u32,
    sparse_scale: f64,
    sparse_reranker_table: Vec<i8>,
    /// Front-coded akshara-id encoding; see codec::encode_vocab.
    vocab_enc: Vec<u8>,
    bigrams: Option<HashMap<String, Vec<(String, u32)>>>,

    // Structural parts of the translit model, unchanged.
    translit_version: u32,
    aksharas: Vec<String>,
    chunks_enc: Vec<u8>,
    trigram_keys_enc: Vec<u8>,

    // Compact numeric sections.
    emissions_enc: Vec<u8>,
    bigrams_lm_enc: Vec<u8>,
    trigrams_enc: Vec<u8>,
    backoff_enc: Vec<u8>,
    unigram_kn_enc: Vec<u8>,
    word_start_enc: Vec<u8>,
    trigram_backoff_enc: Vec<u8>,
}

impl From<&UnifiedModel> for UnifiedModelV3 {
    fn from(m: &UnifiedModel) -> Self {
        let t = &m.translit;
        Self {
            magic: UNIFIED_MAGIC,
            version: UNIFIED_VERSION,
            sparse_scale: m.sparse_scale,
            sparse_reranker_table: m.sparse_reranker_table.clone(),
            vocab_enc: codec::encode_vocab(&m.vocab_freq, &t.aksharas),
            bigrams: m.bigrams.clone(),
            translit_version: t.version,
            aksharas: t.aksharas.clone(),
            chunks_enc: codec::encode_chunks(&t.chunks),
            trigram_keys_enc: codec::encode_pairs(&t.trigram_keys),
            emissions_enc: codec::encode_adjacency(&t.emissions),
            bigrams_lm_enc: codec::encode_adjacency(&t.bigrams),
            trigrams_enc: codec::encode_adjacency(&t.trigrams),
            backoff_enc: codec::encode_weights(&t.backoff),
            unigram_kn_enc: codec::encode_weights(&t.unigram_kn),
            word_start_enc: codec::encode_weights(&t.word_start),
            trigram_backoff_enc: codec::encode_weights(&t.trigram_backoff),
        }
    }
}

impl TryFrom<UnifiedModelV3> for UnifiedModel {
    type Error = Box<dyn std::error::Error>;

    fn try_from(v: UnifiedModelV3) -> Result<Self, Self::Error> {
        let translit = TranslitModel {
            version: v.translit_version,
            aksharas: v.aksharas,
            chunks: codec::decode_chunks(&v.chunks_enc)?,
            emissions: codec::decode_adjacency(&v.emissions_enc)?,
            bigrams: codec::decode_adjacency(&v.bigrams_lm_enc)?,
            backoff: codec::decode_weights(&v.backoff_enc)?,
            unigram_kn: codec::decode_weights(&v.unigram_kn_enc)?,
            word_start: codec::decode_weights(&v.word_start_enc)?,
            trigram_keys: codec::decode_pairs(&v.trigram_keys_enc)?,
            trigrams: codec::decode_adjacency(&v.trigrams_enc)?,
            trigram_backoff: codec::decode_weights(&v.trigram_backoff_enc)?,
            trigram_index: Default::default(),
        };
        let vocab_freq = codec::decode_vocab(&v.vocab_enc, &translit.aksharas)?;
        Ok(Self {
            magic: v.magic,
            version: UNIFIED_VERSION,
            translit,
            sparse_reranker_table: v.sparse_reranker_table,
            sparse_scale: v.sparse_scale,
            vocab_freq,
            bigrams: v.bigrams,
        })
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

    /// The compact encoding must preserve every id exactly and rebuild the
    /// runtime trigram index, which is `#[serde(skip)]` and so is not carried
    /// by any format.
    #[test]
    fn v3_preserves_ngram_structure() {
        let mut m = tiny_model();
        m.translit.bigrams = vec![vec![(0u32, 1.5f32)]];
        m.translit.trigram_keys = vec![(0, 0)];
        m.translit.trigrams = vec![vec![(0u32, 2.5f32)]];
        m.translit.trigram_backoff = vec![0.75];

        let back = UnifiedModel::from_bytes(&m.to_bytes().expect("serialize")).expect("read");
        assert_eq!(back.version, UNIFIED_VERSION);
        assert_eq!(back.translit.trigram_keys, vec![(0, 0)]);
        assert_eq!(back.translit.emissions[0][0].0, 0);
        assert_eq!(back.translit.bigrams[0][0].0, 0);
        assert_eq!(back.translit.trigrams[0][0].0, 0);
        // Few distinct weights, so quantization is exact here.
        assert_eq!(back.translit.trigrams[0][0].1, 2.5);
        assert_eq!(back.translit.trigram_backoff, vec![0.75]);
        // build_trigram_index must have run during load.
        assert_eq!(back.translit.trigram_index.get(&(0, 0)), Some(&0));
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
