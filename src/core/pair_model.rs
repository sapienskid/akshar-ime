// File: src/core/pair_model.rs
//
// Context-dependent pair n-gram grammar (Google Gboard-style WFST
// decoding, Hellsten et al. FSMNLP 2017).
//
// Learns P(chunk_j, akshara_j | chunk_{j-1}, akshara_{j-1}) — a
// bigram over aligned pairs — so conjunct/schwa/matra constraints emerge as
// transitions (a halant pair forces a consonant pair next, etc.).  The model
// is a weighted graph over pair states, decoded by beam search.
// No neural network, no lookup table of words.

use crate::core::translit_model::{pack_chunk_bytes, MAX_CHUNK};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// A (akshara, chunk) pair, packed into a u64 key.
#[inline]
pub fn pair_key(a: u32, chunk: u32) -> u64 {
    ((a as u64) << 32) | chunk as u64
}

#[inline]
pub fn pair_akshara(k: u64) -> u32 {
    (k >> 32) as u32
}

#[inline]
pub fn pair_chunk(k: u64) -> u32 {
    k as u32
}

/// Pair grammar model: bigram over aligned (akshara, chunk) pairs with unigram
/// backoff.  All weights are negative-log probabilities (f32).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairModel {
    pub aksharas: Vec<String>,
    /// -log P(a): marginal akshara unigram (backoff component).
    pub akshara_uni: Vec<f32>,
    /// -log P(s | a) per pair key (emission part; reranker-compatible).
    pub emit_w: HashMap<u64, f32>,
    /// -log P(cur | prev) for observed pair bigrams (KN-smoothed, lower level
    /// folded in at training time).
    pub bi: HashMap<(u64, u64), f32>,
    /// -log P(cur | prev akshara): intermediate backoff level.
    pub bi_ak: HashMap<(u32, u64), f32>,
    /// -log P(cur | prev two pairs): CTW-style deepest context level.
    pub tri: HashMap<((u64, u64), u64), f32>,
    /// -log P(a | word start) (kept for the engine's word-initial handling).
    pub word_start: Vec<f32>,
    /// Dense akshara n-gram LM (word_start prior + KN bigram/trigram over
    /// deterministically segmented corpus words).  The pair grammar is sparse
    /// because EM alignment dilutes transitions; this dense backbone carries
    /// the akshara-sequence statistics (syllable LM) and the pair terms refine it.
    pub ak_lm: AkLm,
}

/// Akshara-level Kneser-Ney tables (subset of TranslitModel).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AkLm {
    pub bigrams: Vec<Vec<(u32, f32)>>,
    pub backoff: Vec<f32>,
    pub unigram_kn: Vec<f32>,
    pub word_start: Vec<f32>,
    pub trigram_keys: Vec<(u32, u32)>,
    pub trigrams: Vec<Vec<(u32, f32)>>,
    pub trigram_backoff: Vec<f32>,
    #[serde(skip, default)]
    pub(crate) trigram_index: HashMap<(u32, u32), usize>,
}

impl AkLm {
    pub fn start_weight(&self, a: u32) -> f64 {
        self.word_start.get(a as usize).copied().unwrap_or(12.0) as f64
    }
    pub fn build_index(&mut self) {
        self.trigram_index = self
            .trigram_keys
            .iter()
            .enumerate()
            .map(|(i, &k)| (k, i))
            .collect();
    }

    pub fn bigram_weight(&self, a: u32, b: u32) -> f64 {
        if let Some(list) = self.bigrams.get(a as usize) {
            if let Some((_, w)) = list.iter().find(|(id, _)| *id == b) {
                return *w as f64;
            }
        }
        let backoff = self.backoff.get(a as usize).copied().unwrap_or(0.0) as f64;
        let uni = self.unigram_kn.get(b as usize).copied().unwrap_or(0.0) as f64;
        backoff + uni
    }

    pub fn trigram_weight(&self, a: u32, b: u32, c: u32) -> f64 {
        if let Some(i) = self.trigram_index.get(&(a, b)) {
            if let Some((_, w)) = self.trigrams[*i].iter().find(|(id, _)| *id == c) {
                return *w as f64;
            }
            let backoff = self.trigram_backoff.get(*i).copied().unwrap_or(0.0) as f64;
            return backoff + self.bigram_weight(b, c);
        }
        self.bigram_weight(b, c)
    }
}

impl PairModel {
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let bytes = bincode::serialize(self).expect("serialize pair model");
        std::fs::write(path, bytes)
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        bincode::deserialize(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Transition weight: -log P(cur | prev) over the 4-level backoff
    /// hierarchy (pair-trigram -> pair-bigram -> akshara-pair-bigram ->
    /// joint pair unigram).  `emit_out` receives the emission -log P(s | a).
    #[inline]
    pub fn transition(&self, prev2: u64, prev: u64, cur: u64, emit_out: &mut f32) -> f32 {
        let a = pair_akshara(cur) as usize;
        let emission = self.emit_w.get(&cur).copied().unwrap_or(12.0);
        *emit_out = emission;
        if prev == 0 {
            return emission;
        }
        if prev2 != 0 {
            if let Some(&w) = self.tri.get(&((prev2, prev), cur)) {
                return w;
            }
        }
        if let Some(&w) = self.bi.get(&(prev, cur)) {
            return w;
        }
        if let Some(&w) = self.bi_ak.get(&(pair_akshara(prev), cur)) {
            return w;
        }
        // Full backoff to the joint unigram P(a) * P(s | a).
        let a_uni = self.akshara_uni.get(a).copied().unwrap_or(14.0);
        emission + a_uni
    }

    /// Reverse index: chunk -> candidate (akshara, emission weight) pairs.
    pub fn reverse_index(&self, max_per_chunk: usize) -> HashMap<u32, Vec<(u32, f32)>> {
        let mut rev: HashMap<u32, Vec<(u32, f32)>> = HashMap::new();
        for (&k, &w) in &self.emit_w {
            rev.entry(pair_chunk(k)).or_default().push((pair_akshara(k), w));
        }
        for v in rev.values_mut() {
            v.sort_by(|a, b| a.1.total_cmp(&b.1));
            v.truncate(max_per_chunk);
        }
        rev
    }

    /// Unfiltered reverse index (diagnostics).
    #[allow(dead_code)]
    pub fn reverse_raw(&self) -> HashMap<u32, Vec<(u32, f32)>> {
        let mut rev: HashMap<u32, Vec<(u32, f32)>> = HashMap::new();
        for (&k, &w) in &self.emit_w {
            rev.entry(pair_chunk(k)).or_default().push((pair_akshara(k), w));
        }
        rev
    }

    /// Decode a packed chunk key back to its roman string (diagnostics).
    pub fn chunk_string(&self, key: u32) -> String {
        crate::core::translit_model::unpack_chunk(key)
    }

    /// All (akshara, weight) candidates for a raw chunk string (diagnostics).
    pub fn chunk_candidates_raw(&self, chunk: &[u8]) -> Vec<(String, f32)> {
        let key = pack_chunk_bytes(chunk);
        let mut v = self.reverse_raw().remove(&key).unwrap_or_default();
        v.sort_by(|a, b| a.1.total_cmp(&b.1));
        v.into_iter()
            .take(6)
            .map(|(a, w)| {
                (
                    self.aksharas.get(a as usize).cloned().unwrap_or_default(),
                    w,
                )
            })
            .collect()
    }
}

/// Beam state for pair decoding.  The merge key must include the *output
/// identity*: paths with the same (pos, prev) context can still spell
/// different strings (न vs ना both consume chunk "na"), so the cheaper
/// spelling must not silently replace the other — otherwise correct
/// candidates are never generated.
#[derive(Debug, Clone)]
struct State {
    pos: usize,
    prev: u64,
    prev2: u64,
    /// Previous akshara id (for the dense akshara LM) and the one before it.
    prev_ak: Option<u32>,
    prev2_ak: Option<u32>,
    emit: f64,
    lm: f64,
    phash: u64,
    path: Vec<u32>,
}

/// splitmix64 finaliser — cheap avalanche mixing for path hashing.
#[inline]
fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ x >> 31
}

#[inline]
fn path_hash(parent: u64, a: u32) -> u64 {
    mix(parent ^ (a as u64).wrapping_add(0x9e3779b97f4a7c15))
}

/// A decoded candidate with decomposed scores (reranker-compatible).
#[derive(Debug, Clone)]
pub struct PairCandidate {
    pub dev: String,
    pub emit: f64,
    pub lm: f64,
    pub akshara_count: usize,
}

pub struct PairDecoder {
    pub model: PairModel,
    reverse: HashMap<u32, Vec<(u32, f32)>>,
    pub lm_weight: f64,
    /// Weight of the pair-grammar context term relative to the akshara LM.
    pub pair_weight: f64,
    beam_width: usize,
}

impl PairDecoder {
    pub fn new(model: PairModel) -> Self {
        let reverse = model.reverse_index(16);
        Self {
            model,
            reverse,
            lm_weight: 1.0,
            pair_weight: 0.5,
            beam_width: 64,
        }
    }

    /// Decode a roman string into k ranked candidates.
    pub fn decode(&self, roman: &str, k: usize) -> Vec<PairCandidate> {
        let roman = roman.to_ascii_lowercase();
        let bytes = roman.as_bytes();
        let m = bytes.len();
        if m == 0 {
            return vec![];
        }
        let k = k.max(1);
        let mut beam = vec![State {
            pos: 0,
            prev: 0,
            prev2: 0,
            prev_ak: None,
            prev2_ak: None,
            emit: 0.0,
            lm: 0.0,
            phash: 0,
            path: Vec::new(),
        }];
        let mut seen: HashMap<String, (f64, f64, f64, usize)> = HashMap::new();

        for _step in 0..32 {
            if beam.is_empty() {
                break;
            }
            let mut next: Vec<State> = Vec::with_capacity(beam.len() * 8);
            for st in &beam {
                let maxl = MAX_CHUNK.min(m - st.pos);
                for l in 1..=maxl {
                    let chunk = pack_chunk_bytes(&bytes[st.pos..st.pos + l]);
                    let Some(cands) = self.reverse.get(&chunk) else {
                        continue;
                    };
                    for &(a, _emit_w0) in cands {
                        let cur = pair_key(a, chunk);
                        let mut emit_w = 0.0f32;
                        let pair_w =
                            self.model.transition(st.prev2, st.prev, cur, &mut emit_w) as f64;
                        // Dense akshara-LM fluency, on top of the
                        // pair-grammar context term.
                        let fluency = match (st.prev2_ak, st.prev_ak) {
                            (_, None) => self.model.ak_lm.start_weight(a),
                            (None, Some(b)) => self.model.ak_lm.bigram_weight(b, a),
                            (Some(a2), Some(b)) => self.model.ak_lm.trigram_weight(a2, b, a),
                        };
                        next.push(State {
                            pos: st.pos + l,
                            prev: cur,
                            prev2: st.prev,
                            prev_ak: Some(a),
                            prev2_ak: st.prev_ak,
                            emit: st.emit + emit_w as f64,
                            lm: st.lm + pair_w * self.pair_weight + fluency,
                            phash: path_hash(st.phash, a),
                            path: {
                                let mut p = st.path.clone();
                                p.push(a);
                                p
                            },
                        });
                    }
                }
            }
            // Dedup on (pos, prev, path-hash): same future context AND same
            // output identity, keeping the best partial score.
            let mut best_by_key: HashMap<(usize, u64, u64), State> = HashMap::new();
            for st in next {
                let sc = st.emit + st.lm * self.lm_weight;
                match best_by_key.get(&(st.pos, st.prev, st.phash)) {
                    Some(bst)
                        if sc >= bst.emit + bst.lm * self.lm_weight => {}
                    _ => {
                        best_by_key.insert((st.pos, st.prev, st.phash), st);
                    }
                }
            }
            let mut merged: Vec<State> = best_by_key.into_values().collect();
            merged.sort_by(|a, b| {
                (a.emit + a.lm * self.lm_weight).total_cmp(&(b.emit + b.lm * self.lm_weight))
            });
            merged.truncate(self.beam_width);
            beam = merged;

            if std::env::var("AKSHAR_V2_DEBUG").is_ok() {
                let pair_str = |k: u64| -> String {
                    let a = self
                        .model
                        .aksharas
                        .get(pair_akshara(k) as usize)
                        .cloned()
                        .unwrap_or_default();
                    format!("{a}+{:?}", self.model.chunk_string(pair_chunk(k)))
                };
                let mut at4: Vec<String> = beam
                    .iter()
                    .filter(|st| st.pos == 4)
                    .take(6)
                    .map(|st| {
                        let dev: String = st
                            .path
                            .iter()
                            .filter_map(|&a| self.model.aksharas.get(a as usize))
                            .map(|s| s.as_str())
                            .collect();
                        format!("{}[{}]={:.2}", dev, pair_str(st.prev), st.emit + st.lm)
                    })
                    .collect();
                at4.sort();
                eprintln!("[step] beam={} pos4: {}", beam.len(), at4.join(" | "));
            }

            // Record complete paths.
            for st in &beam {
                if st.pos == m {
                    let dev: String = st
                        .path
                        .iter()
                        .filter_map(|&a| self.model.aksharas.get(a as usize))
                        .map(|s| s.as_str())
                        .collect();
                    seen.entry(dev).or_insert((
                        st.emit + st.lm * self.lm_weight,
                        st.emit,
                        st.lm,
                        st.path.len(),
                    ));
                }
            }
            if seen.len() >= k * 2 && beam.iter().all(|st| st.pos == m) {
                break;
            }
        }

        let mut out: Vec<PairCandidate> = seen
            .into_iter()
            .map(|(dev, (_score, emit, lm, count))| PairCandidate {
                dev,
                emit,
                lm,
                akshara_count: count,
            })
            .collect();
        out.sort_by(|a, b| {
            (a.emit + a.lm * self.lm_weight).partial_cmp(&(b.emit + b.lm * self.lm_weight))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out.truncate(k);
        out
    }
}
