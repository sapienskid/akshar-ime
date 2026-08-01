// File: src/core/decoder.rs
//
// Generative decoder: ranked Devanagari candidates for a roman string under
// the transliteration model.
//
// The decoder builds a lattice over roman character positions.  Each edge is a
// chunk s of 1..=MAX_CHUNK characters that some akshara a can emit, weighted by
// -log P(s | a) (emission) plus the akshara LM (word-start prior, Kneser-Ney
// bigram/trigram).  Beam search finds the k lowest-total-weight complete paths;
// the concatenated aksharas of each path form a Devanagari candidate string.

use crate::core::translit_model::{TranslitModel, MAX_CHUNK};
use std::collections::HashMap;

const MAX_STEPS: usize = 32;
/// Drop emissions worse than this -log weight (they are alignment noise).
const MAX_EMISSION_WEIGHT: f32 = 8.0;
/// Keep at most this many aksharas per chunk in the reverse index.
const MAX_AKSHARAS_PER_CHUNK: usize = 16;

/// Tunable decoder parameters (exposed for eval-driven tuning).
#[derive(Debug, Clone, Copy)]
pub struct DecoderConfig {
    pub beam_width: usize,
    pub max_emission_weight: f32,
    pub max_aksharas_per_chunk: usize,
    pub lm_weight: f64,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            beam_width: 64,
            max_emission_weight: MAX_EMISSION_WEIGHT,
            max_aksharas_per_chunk: MAX_AKSHARAS_PER_CHUNK,
            lm_weight: 1.0,
        }
    }
}

/// A single lattice edge: consumes `len` roman chars, emits akshara `a` at weight `w`.
#[derive(Debug, Clone, Copy)]
struct Edge {
    len: usize,
    a: u32,
    w: f32,
}

pub struct ModelDecoder {
    pub model: TranslitModel,
    /// Reverse index: chunk -> (akshara, weight), capped + sorted by weight.
    reverse: HashMap<String, Vec<(u32, f32)>>,
    /// Search / scoring configuration.
    pub config: DecoderConfig,
}

#[derive(Debug, Clone)]
struct BeamState {
    pos: usize,
    prev: Option<u32>, // last akshara (None = word start)
    prev2: Option<u32>, // second-to-last akshara
    score: f64,         // total = emit + lm_weight * lm (for beam ordering)
    emit: f64,          // accumulated emission -log weight
    lm: f64,            // accumulated LM -log weight (unscaled)
    path: Vec<u32>,
}

/// A decoded candidate with its decomposed scores, for discriminative reranking.
#[derive(Debug, Clone)]
pub struct DecodedCandidate {
    pub dev: String,
    /// Sum of emission weights (-log P(R | aksharas), best alignment).
    pub emit: f64,
    /// Sum of LM weights (word-start + bigram + trigram), unscaled.
    pub lm: f64,
    /// Number of aksharas in the candidate.
    pub akshara_count: usize,
}

impl ModelDecoder {
    pub fn new(model: TranslitModel) -> Self {
        Self::with_config(model, DecoderConfig::default())
    }

    /// Build a decoder with explicit search/scoring parameters.
    pub fn with_config(model: TranslitModel, config: DecoderConfig) -> Self {
        // Build the reverse index: chunk -> (akshara, weight), capped + sorted.
        let mut reverse: HashMap<String, Vec<(u32, f32)>> = HashMap::new();
        for (a, list) in model.emissions.iter().enumerate() {
            for &(cid, w) in list {
                if w >= config.max_emission_weight {
                    continue;
                }
                if let Some(chunk) = model.chunks.get(cid as usize) {
                    reverse
                        .entry(chunk.clone())
                        .or_insert_with(Vec::new)
                        .push((a as u32, w));
                }
            }
        }
        for v in reverse.values_mut() {
            v.sort_by(|a, b| a.1.total_cmp(&b.1));
            v.truncate(config.max_aksharas_per_chunk);
        }
        Self { model, reverse, config }
    }

    /// Configure the LM weight relative to emission weights (default 1.0).
    pub fn with_lm_weight(mut self, w: f64) -> Self {
        self.config.lm_weight = w;
        self
    }

    /// Expose the reverse-index candidates for a chunk (diagnostics / search).
    pub fn chunk_candidates(&self, chunk: &str) -> &[(u32, f32)] {
        self.reverse.get(chunk).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Decode a roman string into `k` ranked (devanagari, -log score) pairs.
    pub fn decode(&self, roman: &str, k: usize) -> Vec<(String, f64)> {
        self.decode_detailed(roman, k)
            .into_iter()
            .map(|c| (c.dev, c.emit + c.lm * self.config.lm_weight))
            .collect()
    }

    /// Decode into decomposed candidates (emission + LM separately) for reranking.
    pub fn decode_detailed(&self, roman: &str, k: usize) -> Vec<DecodedCandidate> {
        let roman = roman.to_ascii_lowercase();
        let edges_by_pos = self.build_edges(&roman);
        let m = roman.len();
        if m == 0 {
            return vec![];
        }
        let k = k.max(1);

        let mut beam: Vec<BeamState> = vec![BeamState {
            pos: 0,
            prev: None,
            prev2: None,
            score: 0.0,
            emit: 0.0,
            lm: 0.0,
            path: Vec::new(),
        }];
        // Complete paths: path -> (score, emit, lm), kept min by score.
        let mut seen: HashMap<Vec<u32>, (f64, f64, f64)> = HashMap::new();

        for _step in 0..MAX_STEPS {
            if beam.is_empty() {
                break;
            }
            let mut next: Vec<BeamState> = Vec::with_capacity(beam.len() * 16);

            for st in &beam {
                if st.pos == m {
                    seen.entry(st.path.clone())
                        .and_modify(|best| {
                            if st.score < best.0 {
                                *best = (st.score, st.emit, st.lm);
                            }
                        })
                        .or_insert((st.score, st.emit, st.lm));
                    continue;
                }
                for &e in &edges_by_pos[st.pos] {
                    let fluency = match (st.prev2, st.prev) {
                        (_, None) => self.model.start_weight(e.a),
                        (None, Some(b)) => self.model.bigram_weight(b, e.a),
                        (Some(a), Some(b)) => self.model.trigram_weight(a, b, e.a),
                    };
                    let emit = st.emit + e.w as f64;
                    let lm = st.lm + fluency;
                    let score = emit + lm * self.config.lm_weight;
                    let mut path = st.path.clone();
                    path.push(e.a);
                    next.push(BeamState {
                        pos: st.pos + e.len,
                        prev: Some(e.a),
                        prev2: st.prev,
                        score,
                        emit,
                        lm,
                        path,
                    });
                }
            }

            // Dedup by (pos, path) keeping best score.
            let mut best_by_key: HashMap<(usize, Vec<u32>), (f64, f64, f64)> = HashMap::new();
            for cand in next {
                best_by_key
                    .entry((cand.pos, cand.path.clone()))
                    .and_modify(|best| {
                        if cand.score < best.0 {
                            *best = (cand.score, cand.emit, cand.lm);
                        }
                    })
                    .or_insert((cand.score, cand.emit, cand.lm));
            }
            let mut deduped: Vec<BeamState> = best_by_key
                .into_iter()
                .map(|((pos, path), (score, emit, lm))| {
                    let prev = path.last().copied();
                    let prev2 = path.len().checked_sub(2).and_then(|i| path.get(i)).copied();
                    BeamState { pos, prev, prev2, score, emit, lm, path }
                })
                .collect();
            deduped.sort_by(|a, b| a.score.total_cmp(&b.score));
            deduped.truncate(self.config.beam_width);
            beam = deduped;
        }

        let mut results: Vec<DecodedCandidate> = seen
            .into_iter()
            .map(|(path, (_, emit, lm))| DecodedCandidate {
                dev: self.path_to_string(&path),
                emit,
                lm,
                akshara_count: path.len(),
            })
            .collect();
        results.sort_by(|a, b| {
            let at = a.emit + a.lm * self.config.lm_weight;
            let bt = b.emit + b.lm * self.config.lm_weight;
            at.total_cmp(&bt)
        });
        results.truncate(k);
        results
    }

    fn build_edges(&self, roman: &str) -> Vec<Vec<Edge>> {
        let m = roman.len();
        let mut edges = vec![Vec::new(); m + 1];
        for pos in 0..m {
            for l in 1..=MAX_CHUNK.min(m - pos) {
                let chunk = &roman[pos..pos + l];
                if let Some(list) = self.reverse.get(chunk) {
                    for &(a, w) in list {
                        edges[pos].push(Edge { len: l, a, w });
                    }
                }
            }
        }
        edges
    }

    fn path_to_string(&self, path: &[u32]) -> String {
        let mut out = String::with_capacity(path.len() * 3);
        for &a in path {
            if let Some(s) = self.model.aksharas.get(a as usize) {
                out.push_str(s);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::em_trainer::{Trainer, TrainerConfig};

    fn trained_model(pairs: &[(&str, &str)]) -> TranslitModel {
        let mut t = Trainer::new();
        for (r, d) in pairs {
            t.add_pair(r, d);
        }
        t.finalize(&TrainerConfig { iterations: 6, ..Default::default() })
    }

    #[test]
    fn decodes_single_akshara() {
        let model = trained_model(&[("ka", "क"), ("ki", "कि"), ("ku", "कु")]);
        let dec = ModelDecoder::new(model);
        let res = dec.decode("ka", 5);
        assert!(!res.is_empty());
        assert_eq!(res[0].0, "क");
    }

    #[test]
    fn decodes_word_with_matra() {
        let model = trained_model(&[("ka", "क"), ("ki", "कि"), ("kama", "कम"), ("nama", "नम")]);
        let dec = ModelDecoder::new(model);
        let res = dec.decode("kama", 5);
        assert!(!res.is_empty());
        assert_eq!(res[0].0, "कम");
    }

    #[test]
    fn decodes_namaste() {
        let model = trained_model(&[("namaste", "नमस्ते"), ("nama", "नम"), ("nepal", "नेपाल")]);
        let dec = ModelDecoder::new(model);
        let res = dec.decode("namaste", 8);
        assert!(!res.is_empty());
        assert!(res.iter().any(|(d, _)| d == "नमस्ते"));
        assert_eq!(res[0].0, "नमस्ते");
    }

    #[test]
    fn decodes_handles_empty() {
        let model = trained_model(&[("ka", "क")]);
        let dec = ModelDecoder::new(model);
        assert!(dec.decode("", 5).is_empty());
    }
}
