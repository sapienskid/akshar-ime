// File: src/core/em_trainer.rs
//
// EM trainer for the generative transliteration model.
//
// Following Li, Zhang & Su (ACL-04), we learn the transliteration table
// P(chunk | akshara) from an unaligned parallel lexicon.  Each Devanagari word
// is segmented into aksharas; the roman word is a sequence of chars.  A
// segmentation of the roman word assigns each akshara a contiguous chunk of
// 0..=MAX_CHUNK roman characters.  EM (forward-backward over the alignment
// space) maximises the corpus likelihood under
//
//   P(R | D) = sum_segmentations prod_j P(chunk_j | akshara_j).
//
// Emissions are seeded from a deterministic codepoint aligner (`alignment.rs`)
// so EM starts from a sensible point, then refined with a few EM iterations.
// A Kneser-Ney akshara n-gram LM is built from the same corpus for scoring
// fresh transliterations.

use crate::core::akshara::segment;
use crate::core::alignment::align_emissive;
use crate::core::translit_model::{
    pack_chunk, pack_chunk_bytes, unpack_chunk, TranslitModel, MAX_CHUNK,
};
use crate::core::v2::{pair_akshara, pair_key, AkLm, PairModel};
use std::collections::{BTreeSet, HashMap};

/// A single training pair held in memory: roman bytes + akshara id sequence.
struct Pair {
    roman: Vec<u8>,
    aks: Vec<u32>,
    /// Observation weight (frequency): scales LM counts and EM posteriors.
    weight: f64,
}

pub struct TrainerConfig {
    /// Number of EM passes over the corpus.
    pub iterations: usize,
    /// Add-alpha smoothing on the emission M-step (keeps probabilities > 0).
    pub em_smoothing: f64,
    /// Kneser-Ney absolute discount for the akshara bigram LM.
    pub kn_discount: f64,
    /// Seed emissions from the deterministic codepoint aligner before EM.
    pub seed_from_aligner: bool,
    /// Stop ingesting after this many clean pairs (iterate fast in dev).
    pub limit: Option<usize>,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            iterations: 12,
            em_smoothing: 0.05,
            kn_discount: 0.75,
            seed_from_aligner: true,
            limit: None,
        }
    }
}

pub struct Trainer {
    akshara_map: HashMap<String, u32>,
    akshara_list: Vec<String>,
    pairs: Vec<Pair>,
    limit: Option<usize>,

    // Seed counts (raw) and live emission probabilities.
    seed_counts: Vec<HashMap<u32, f64>>,
    emission: Vec<HashMap<u32, f64>>,

    // LM counts.
    unigram_counts: Vec<u64>,
    word_initial: Vec<u64>,
    total_words: u64,
    bigram_counts: HashMap<(u32, u32), u64>,
    trigram_counts: HashMap<(u32, u32, u32), u64>,
    /// continuation[akshara] = number of distinct left-context aksharas.
    continuation: Vec<u64>,
    /// successor_count[(a,b)] = number of distinct c with (a,b,c) seen.
    trigram_successors: HashMap<(u32, u32), u64>,

    // Global chunk unigram (for Dirichlet-smoothed emissions).
    chunk_unigram: HashMap<u32, f64>,
    total_chunk_obs: f64,

    distinct_bigrams: u64,
    pub ingested: usize,
    pub skipped: usize,
}

impl Trainer {
    pub fn new() -> Self {
        Self {
            akshara_map: HashMap::new(),
            akshara_list: Vec::new(),
            pairs: Vec::new(),
            limit: None,
            seed_counts: Vec::new(),
            emission: Vec::new(),
            unigram_counts: Vec::new(),
            word_initial: Vec::new(),
            total_words: 0,
            bigram_counts: HashMap::new(),
            trigram_counts: HashMap::new(),
            continuation: Vec::new(),
            trigram_successors: HashMap::new(),
            chunk_unigram: HashMap::new(),
            total_chunk_obs: 0.0,
            distinct_bigrams: 0,
            ingested: 0,
            skipped: 0,
        }
    }

    /// Builder: cap the number of clean pairs ingested (dev/testing).
    pub fn with_limit(mut self, limit: Option<usize>) -> Self {
        self.limit = limit;
        self
    }

    fn intern_akshara(&mut self, akshara: &str) -> u32 {
        if let Some(&id) = self.akshara_map.get(akshara) {
            return id;
        }
        let id = self.akshara_list.len() as u32;
        self.akshara_list.push(akshara.to_string());
        self.akshara_map.insert(akshara.to_string(), id);
        self.seed_counts.push(HashMap::new());
        self.emission.push(HashMap::new());
        self.unigram_counts.push(0);
        self.word_initial.push(0);
        self.continuation.push(0);
        id
    }

    /// Add one (roman, devanagari) pair.
    ///
    /// All well-formed devanagari words contribute akshara LM counts.  Pairs
    /// whose roman is clean lowercase ASCII `a-z` and of sane length also feed
    /// the EM emission model (seeded from the codepoint aligner).
    pub fn add_pair(&mut self, roman: &str, dev: &str) {
        self.add_pair_weighted(roman, dev, 1.0);
    }

    /// Add a pair with an observation weight (e.g. corpus frequency of a
    /// synthetic pair).  Weight scales the LM counts and the EM posteriors.
    pub fn add_pair_weighted(&mut self, roman: &str, dev: &str, weight: f64) {
        if roman.is_empty() || dev.is_empty() {
            self.skipped += 1;
            return;
        }

        let aks: Vec<u32> = segment(dev)
            .iter()
            .map(|a| self.intern_akshara(a))
            .collect();
        if aks.is_empty() {
            self.skipped += 1;
            return;
        }

        // LM counts (independent of roman quality), scaled by weight.
        for &a in &aks {
            self.unigram_counts[a as usize] += weight as u64;
        }
        self.word_initial[aks[0] as usize] += weight as u64;
        self.total_words += weight as u64;
        for w in aks.windows(2) {
            let (b, c) = (w[0], w[1]);
            let e = self.bigram_counts.entry((b, c)).or_insert(0);
            if *e == 0 {
                self.continuation[c as usize] += 1;
                self.distinct_bigrams += 1;
            }
            *e += weight as u64;
        }
        for w in aks.windows(3) {
            let (a, b, c) = (w[0], w[1], w[2]);
            let e = self.trigram_counts.entry((a, b, c)).or_insert(0);
            if *e == 0 {
                *self.trigram_successors.entry((a, b)).or_insert(0) += 1;
            }
            *e += weight as u64;
        }

        // EM pairs: clean lowercase a-z roman, bounded lengths.
        let roman_ok = !roman.is_empty()
            && roman.len() <= 24
            && aks.len() <= 12
            && roman.bytes().all(|b| b.is_ascii_lowercase());
        if !roman_ok {
            self.skipped += 1;
            return;
        }
        if let Some(limit) = self.limit {
            if self.ingested >= limit {
                return;
            }
        }

        // Seed the emission counts from the codepoint aligner, grouped by akshara.
        // Bare-consonant aksharas also get their inherent-schwa variant (`s` + "a")
        // and schwa-dropped variant (drop a trailing "a") so EM can discover the
        // medial-schwa convention (e.g. "cha" -> च).
        let aligned = align_emissive(roman, dev);
        let mut p = 0usize;
        for &a in &aks {
            let aks_str = self.akshara_list[a as usize].clone();
            let aks_len = aks_str.chars().count();
            let mut chunk = String::with_capacity(2);
            for _ in 0..aks_len {
                if let Some(pair) = aligned.get(p) {
                    chunk.push_str(&pair.roman);
                    p += 1;
                }
            }
            if !chunk.is_empty() && chunk.len() <= MAX_CHUNK {
                self.seed_chunk(a, &chunk);
                if is_bare_consonant(&aks_str) {
                    if chunk.len() < MAX_CHUNK {
                        let mut with_a = chunk.clone();
                        with_a.push('a');
                        self.seed_chunk(a, &with_a);
                    }
                    if chunk.ends_with('a') && chunk.len() > 1 {
                        self.seed_chunk(a, &chunk[..chunk.len() - 1]);
                    }
                }
            }
        }

        let id = self.ingested;
        if self.pairs.len() <= id {
            self.pairs.push(Pair {
                roman: roman.as_bytes().to_vec(),
                aks,
                weight,
            });
        } else {
            self.pairs[id] = Pair {
                roman: roman.as_bytes().to_vec(),
                aks,
                weight,
            };
        }
        self.ingested += 1;
    }

    fn seed_chunk(&mut self, a: u32, chunk: &str) {
        let key = pack_chunk(chunk);
        *self.seed_counts[a as usize].entry(key).or_insert(0.0) += 1.0;
        *self.chunk_unigram.entry(key).or_insert(0.0) += 1.0;
        self.total_chunk_obs += 1.0;
    }

    /// Finalise: initialise emissions, run EM, build the akshara LM, serialise-ready model.
    pub fn finalize(&mut self, config: &TrainerConfig) -> TranslitModel {
        self.init_emissions(config.seed_from_aligner);
        self.run_em(config.iterations, config.em_smoothing);
        let mut model = TranslitModel {
            version: crate::core::translit_model::MODEL_VERSION,
            ..Default::default()
        };
        model.aksharas = self.akshara_list.clone();

        // Chunk vocabulary from emission keys, sorted for determinism.
        let mut chunk_set: BTreeSet<u32> = BTreeSet::new();
        for em in &self.emission {
            for &k in em.keys() {
                chunk_set.insert(k);
            }
        }
        let mut chunk_id_map: HashMap<u32, u32> = HashMap::new();
        for (i, &k) in chunk_set.iter().enumerate() {
            chunk_id_map.insert(k, i as u32);
            model.chunks.push(unpack_chunk(k));
        }

        model.emissions = self
            .emission
            .iter()
            .map(|em| {
                let mut v: Vec<(u32, f32)> = em
                    .iter()
                    .map(|(k, p)| {
                        let w = if *p > 0.0 { -p.ln() as f32 } else { 50.0 };
                        (chunk_id_map[k], w)
                    })
                    .collect();
                v.sort_by_key(|(cid, _)| *cid);
                v
            })
            .collect();

        self.build_kn_lm(&mut model, config.kn_discount);
        model
    }

    fn init_emissions(&mut self, seed_from_aligner: bool) {
        if seed_from_aligner {
            for a in 0..self.akshara_list.len() {
                let counts = std::mem::take(&mut self.seed_counts[a]);
                if counts.is_empty() {
                    continue;
                }
                let total: f64 = counts.values().sum();
                let mut em: HashMap<u32, f64> = HashMap::with_capacity(counts.len());
                for (k, c) in counts {
                    // Floor small counts so the first E-step isn't degenerate.
                    em.insert(k, (c / total).max(1e-4));
                }
                self.emission[a] = em;
            }
            return;
        }
        // Fallback init (no aligner seed): uniform over observed chunks.
        // The first EM iteration will still learn from the data.
        for a in 0..self.akshara_list.len() {
            let counts = std::mem::take(&mut self.seed_counts[a]);
            let mut em: HashMap<u32, f64> = HashMap::new();
            for k in counts.keys() {
                em.insert(*k, 1.0);
            }
            self.emission[a] = em;
        }
    }

    fn build_kn_lm(&self, model: &mut TranslitModel, delta: f64) {
        let n = self.akshara_list.len();
        let distinct = self.distinct_bigrams as f64;
        let mut unigram_kn = vec![0.0f32; n];
        for (a, w) in unigram_kn.iter_mut().enumerate() {
            let cont = self.continuation[a] as f64;
            // Floor so word-initial-only aksharas keep a finite log-prob.
            let p = (cont + 0.5) / (distinct + 0.5 * n as f64);
            *w = -p.ln() as f32;
        }

        let mut by_left: HashMap<u32, Vec<(u32, u64)>> = HashMap::new();
        for (&(b, c), &cnt) in &self.bigram_counts {
            by_left.entry(b).or_default().push((c, cnt));
        }

        let mut bigrams = vec![Vec::new(); n];
        let mut backoff = vec![0.0f32; n];
        for a in 0..n {
            let Some(list) = by_left.get(&(a as u32)) else {
                // No outgoing bigrams observed: pure backoff to the unigram.
                backoff[a] = 0.0;
                continue;
            };
            let total: u64 = list.iter().map(|(_, c)| c).sum();
            let n_distinct = list.len() as f64;
            let lambda = delta * n_distinct / total as f64;
            backoff[a] = (-lambda.ln()) as f32;
            let mut v = Vec::with_capacity(list.len());
            for &(c, cnt) in list {
                let disc = (cnt as f64 - delta).max(0.0) / total as f64;
                let p_kn_c = (-unigram_kn[c as usize] as f64).exp();
                let p = disc + lambda * p_kn_c;
                let w = if p > 0.0 { -p.ln() } else { 50.0 };
                v.push((c, w as f32));
            }
            v.sort_by_key(|(id, _)| *id);
            bigrams[a] = v;
        }
        model.bigrams = bigrams;
        model.backoff = backoff;
        model.unigram_kn = unigram_kn;

        // Word-start prior: P(a | word start) from corpus word-initial counts.
        let mut word_start = vec![0.0f32; n];
        for (a, w) in word_start.iter_mut().enumerate() {
            let c = self.word_initial[a] as f64;
            // Floor keeps log finite for aksharas that never start words.
            let p = (c + 0.5) / (self.total_words as f64 + 0.5 * n as f64);
            *w = -p.ln() as f32;
        }
        model.word_start = word_start;

        // Trigram KN LM: group counts by (a,b) context.
        let mut by_ctx: HashMap<(u32, u32), Vec<(u32, u64)>> = HashMap::new();
        for (&(a, b, c), &cnt) in &self.trigram_counts {
            by_ctx.entry((a, b)).or_default().push((c, cnt));
        }
        let mut ctxs: Vec<(u32, u32)> = by_ctx.keys().cloned().collect();
        ctxs.sort();
        let mut trigram_keys = Vec::with_capacity(ctxs.len());
        let mut trigrams = Vec::with_capacity(ctxs.len());
        let mut trigram_backoff = Vec::with_capacity(ctxs.len());
        for &(a, b) in &ctxs {
            let list = &by_ctx[&(a, b)];
            let c_ab = self.bigram_counts.get(&(a, b)).copied().unwrap_or(1) as f64;
            let a_succ = self.trigram_successors.get(&(a, b)).copied().unwrap_or(1) as f64;
            let lambda = delta * a_succ / c_ab;
            trigram_backoff.push((-lambda.ln()) as f32);
            let mut v = Vec::with_capacity(list.len());
            for &(c, cnt) in list {
                let disc = (cnt as f64 - delta).max(0.0) / c_ab;
                let p_kn_c = (-model.bigram_weight(b, c)).exp();
                let p = disc + lambda * p_kn_c;
                let w = if p > 0.0 { -p.ln() } else { 50.0 };
                v.push((c, w as f32));
            }
            v.sort_by_key(|(id, _)| *id);
            trigram_keys.push((a, b));
            trigrams.push(v);
        }
        model.trigram_keys = trigram_keys;
        model.trigrams = trigrams;
        model.trigram_backoff = trigram_backoff;
        model.build_trigram_index();
    }

    fn run_em(&mut self, iterations: usize, alpha: f64) {
        if self.pairs.is_empty() {
            return;
        }
        for it in 0..iterations {
            self.em_iteration(alpha);
            if it % 3 == 2 {
                eprintln!("  [em] iteration {}/{} done", it + 1, iterations);
            }
        }
    }

    fn em_iteration(&mut self, alpha: f64) {
        let n_aksharas = self.akshara_list.len();
        let mut counts: Vec<HashMap<u32, f64>> = (0..n_aksharas).map(|_| HashMap::new()).collect();

        // The E-step is embarrassingly parallel over pairs: each thread computes
        // local posterior counts for a slice, then we merge.  The emission maps
        // and pairs are read-only, so scoped threads can share them safely.
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(16);
        let pairs = &self.pairs;
        let emission = &self.emission;
        if !pairs.is_empty() && threads > 1 {
            let chunk_size = pairs.len().div_ceil(threads);
            std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(threads);
                for chunk in pairs.chunks(chunk_size) {
                    handles.push(s.spawn(move || e_step_chunk(chunk, emission, n_aksharas)));
                }
                for h in handles {
                    for (a, map) in h
                        .join()
                        .expect("e-step thread panicked")
                        .into_iter()
                        .enumerate()
                    {
                        let ca = &mut counts[a];
                        for (k, v) in map {
                            *ca.entry(k).or_insert(0.0) += v;
                        }
                    }
                }
            });
        } else {
            let local = e_step_chunk(pairs, emission, n_aksharas);
            for (a, map) in local.into_iter().enumerate() {
                counts[a] = map;
            }
        }

        // --- M-step: normalise + Dirichlet-smooth toward the global chunk unigram. ---
        let total_chunk = self.total_chunk_obs.max(1.0);
        for (a, counts_a) in counts.iter().enumerate() {
            if counts_a.is_empty() {
                continue;
            }
            let total: f64 = counts_a.values().sum();
            let denom = total + alpha;
            let mut new_em: HashMap<u32, f64> = HashMap::with_capacity(counts_a.len());
            for (k, c) in counts_a {
                let prior = self.chunk_unigram.get(k).copied().unwrap_or(0.0) / total_chunk;
                new_em.insert(*k, (c + alpha * prior) / denom);
            }
            self.emission[a] = new_em;
        }
    }
}

/// Forward-backward EM E-step over a slice of pairs.
///
/// Returns per-akshara fractional posterior counts of (packed chunk -> count).
/// This is a free function so it can run on scoped threads sharing the emission
/// tables read-only.
fn e_step_chunk(
    pairs: &[Pair],
    emission: &[HashMap<u32, f64>],
    n_aksharas: usize,
) -> Vec<HashMap<u32, f64>> {
    let mut counts: Vec<HashMap<u32, f64>> = (0..n_aksharas).map(|_| HashMap::new()).collect();
    let mut f: Vec<Vec<f64>> = Vec::new();
    let mut b: Vec<Vec<f64>> = Vec::new();

    for pair in pairs {
        let m = pair.roman.len();
        let n = pair.aks.len();
        if m == 0 || n == 0 {
            continue;
        }

        // --- Forward pass. ---
        f.clear();
        f.resize(n + 1, vec![0.0f64; m + 1]);
        f[0][0] = 1.0;
        for j in 1..=n {
            let em = &emission[pair.aks[j - 1] as usize];
            let prev = f[j - 1].clone();
            let mut row = vec![0.0f64; m + 1];
            for i in 0..=m {
                let maxl = MAX_CHUNK.min(i);
                let mut acc = 0.0;
                for l in 0..=maxl {
                    let key = pack_chunk_bytes(&pair.roman[i - l..i]);
                    if let Some(&p) = em.get(&key) {
                        acc += prev[i - l] * p;
                    }
                }
                row[i] = acc;
            }
            f[j] = row;
        }
        let z = f[n][m];
        // Sub-normal z (model assigns near-zero total probability) makes
        // inv_z explode and poisons every transition count with ~1e300 junk;
        // skip such words entirely.
        if z < 1e-12 || !z.is_finite() {
            continue;
        }

        // --- Backward pass. ---
        b.clear();
        b.resize(n + 1, vec![0.0f64; m + 1]);
        b[n][m] = 1.0;
        for j in (0..n).rev() {
            let em = &emission[pair.aks[j] as usize];
            let next = b[j + 1].clone();
            let mut row = vec![0.0f64; m + 1];
            for i in 0..=m {
                let maxl = MAX_CHUNK.min(m - i);
                let mut acc = 0.0;
                for l in 0..=maxl {
                    let key = pack_chunk_bytes(&pair.roman[i..i + l]);
                    if let Some(&p) = em.get(&key) {
                        acc += p * next[i + l];
                    }
                }
                row[i] = acc;
            }
            b[j] = row;
        }

        // --- Accumulate posteriors. ---
        let inv_z = 1.0 / z;
        for (idx, &a) in pair.aks.iter().enumerate() {
            let j = idx + 1;
            let em = &emission[a as usize];
            let counts_a = &mut counts[a as usize];
            for i in 1..=m {
                let maxl = MAX_CHUNK.min(i);
                for l in 1..=maxl {
                    let key = pack_chunk_bytes(&pair.roman[i - l..i]);
                    if let Some(&p) = em.get(&key) {
                        let post = f[j - 1][i - l] * p * b[j][i] * inv_z;
                        if post > 0.0 {
                            *counts_a.entry(key).or_insert(0.0) += post * pair.weight;
                        }
                    }
                }
            }
            if let Some(&p0) = em.get(&0u32) {
                if p0 > 0.0 {
                    for i in 0..=m {
                        let post = f[j - 1][i] * p0 * b[j][i] * inv_z;
                        if post > 0.0 {
                            *counts_a.entry(0u32).or_insert(0.0) += post * pair.weight;
                        }
                    }
                }
            }
        }
    }
    counts
}

impl Default for Trainer {
    fn default() -> Self {
        Self::new()
    }
}

/// Posterior mass over consecutive aligned pairs: returns
/// (prev_pair_key, cur_pair_key, posterior) triples.  The unigram posterior is
/// the ordinary `e_step_chunk` count, so only transitions are collected here.
fn pair_transitions(
    pairs: &[Pair],
    emission: &[HashMap<u32, f64>],
    _n_aksharas: usize,
) -> (Vec<(u64, u64, u32, f64)>, Vec<((u64, u64), u64, f64)>) {
    let mut out: Vec<(u64, u64, u32, f64)> = Vec::new();
    // Depth-2 pair context: ((pair_{j-2}, pair_{j-1}) -> pair_j) posteriors.
    let mut out2: Vec<((u64, u64), u64, f64)> = Vec::new();
    let mut f: Vec<Vec<f64>> = Vec::new();
    let mut b: Vec<Vec<f64>> = Vec::new();

    for pair in pairs {
        let m = pair.roman.len();
        let n = pair.aks.len();
        if m == 0 || n < 2 {
            continue;
        }

        // Forward pass (identical to e_step_chunk).
        f.clear();
        f.resize(n + 1, vec![0.0f64; m + 1]);
        f[0][0] = 1.0;
        for j in 1..=n {
            let em = &emission[pair.aks[j - 1] as usize];
            let prev = f[j - 1].clone();
            let mut row = vec![0.0f64; m + 1];
            for i in 0..=m {
                let maxl = MAX_CHUNK.min(i);
                let mut acc = 0.0;
                for l in 0..=maxl {
                    let key = pack_chunk_bytes(&pair.roman[i - l..i]);
                    if let Some(&p) = em.get(&key) {
                        acc += prev[i - l] * p;
                    }
                }
                row[i] = acc;
            }
            f[j] = row;
        }
        let z = f[n][m];
        // Sub-normal z (model assigns near-zero total probability) makes
        // inv_z explode and poisons every transition count with ~1e300 junk;
        // skip such words entirely.
        if z < 1e-12 || !z.is_finite() {
            continue;
        }

        // Backward pass (identical to e_step_chunk).
        b.clear();
        b.resize(n + 1, vec![0.0f64; m + 1]);
        b[n][m] = 1.0;
        for j in (0..n).rev() {
            let em = &emission[pair.aks[j] as usize];
            let next = b[j + 1].clone();
            let mut row = vec![0.0f64; m + 1];
            for i in 0..=m {
                let maxl = MAX_CHUNK.min(m - i);
                let mut acc = 0.0;
                for l in 0..=maxl {
                    let key = pack_chunk_bytes(&pair.roman[i..i + l]);
                    if let Some(&p) = em.get(&key) {
                        acc += p * next[i + l];
                    }
                }
                row[i] = acc;
            }
            b[j] = row;
        }

        // Transition posteriors: aksharas j-1 -> j, chunks split at i1.
        let inv_z = 1.0 / z;
        for j in 1..n {
            let a1 = pair.aks[j - 1];
            let a2 = pair.aks[j];
            let em1 = &emission[a1 as usize];
            let em2 = &emission[a2 as usize];
            for i1 in 1..m {
                let maxl1 = MAX_CHUNK.min(i1);
                for l1 in 1..=maxl1 {
                    let key1 = pack_chunk_bytes(&pair.roman[i1 - l1..i1]);
                    let Some(&p1) = em1.get(&key1) else {
                        continue;
                    };
                    if p1 <= 0.0 {
                        continue;
                    }
                    let maxl2 = MAX_CHUNK.min(m - i1);
                    for l2 in 1..=maxl2 {
                        let key2 = pack_chunk_bytes(&pair.roman[i1..i1 + l2]);
                        let Some(&p2) = em2.get(&key2) else {
                            continue;
                        };
                        if p2 <= 0.0 {
                            continue;
                        }
                        // Suffix AFTER the cur pair: 1-based b index j+2; when
                        // the cur pair is the last akshara the suffix is the
                        // boundary condition (must end exactly at m).
                        let bval = if j + 2 <= n {
                            b[j + 2][i1 + l2]
                        } else if i1 + l2 == m {
                            1.0
                        } else {
                            0.0
                        };
                        let post = f[j - 1][i1 - l1] * p1 * p2 * bval * inv_z;
                        // A true posterior is <= 1; degenerate words (tiny z)
                        // can numerically violate this and poison counts.
                        if post >= 1e-6 {
                            let prev1 = pair_key(a1, key1);
                            let cur = pair_key(a2, key2);
                            out.push((prev1, cur, a1, post.min(1.0)));
                            // Depth-2: enumerate the split of akshara j-2's chunk.
                            if j >= 2 {
                                let a0 = pair.aks[j - 2];
                                let em0 = &emission[a0 as usize];
                                let f0 = f[j - 2][i1 - l1];
                                if f0 > 0.0 {
                                    for l0 in 1..=MAX_CHUNK.min(i1 - l1) {
                                        let key0 =
                                            pack_chunk_bytes(&pair.roman[i1 - l1 - l0..i1 - l1]);
                                        if let Some(&p0) = em0.get(&key0) {
                                            if p0 > 0.0 {
                                                let post2 = f[j - 2][i1 - l1 - l0]
                                                    * p0
                                                    * p1
                                                    * p2
                                                    * bval
                                                    * inv_z;
                                                if post2 >= 1e-6 {
                                                    out2.push((
                                                        (
                                                            pair_key(a0, key0),
                                                            prev1,
                                                        ),
                                                        cur,
                                                        post2.min(1.0),
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (out, out2)
}

impl Trainer {
    /// M1: train the v2 pair-bigram grammar.  Runs the same EM loop as
    /// `finalize` (identical unigram emissions — they become the pair unigram
    /// backoff), then accumulates posteriors over consecutive aligned pairs
    /// and normalises them into P(cur | prev) with unigram backoff.
    pub fn finalize_pair(&mut self, config: &TrainerConfig, alpha_pair: f64) -> PairModel {
        self.init_emissions(config.seed_from_aligner);
        self.run_em(config.iterations, config.em_smoothing);

        // --- Collect bigram transition posteriors with the final emissions. ---
        eprintln!("  [v2] collecting pair-bigram transitions...");
        let n_aksharas = self.akshara_list.len();
        let pairs = &self.pairs;
        let emission = &self.emission;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(16);
        let mut bi: HashMap<(u64, u64), f64> = HashMap::new();
        // Intermediate backoff level: (prev akshara, cur pair) — much denser
        // than full pair context, carries genuine akshara bigram structure.
        let mut bi_ak: HashMap<(u32, u64), f64> = HashMap::new();
        // Depth-2 pair context (CTW-style deepest level), flat 3-tuple keys.
        let mut tri: HashMap<((u64, u64), u64), f64> = HashMap::new();
        if !pairs.is_empty() && threads > 1 {
            let chunk_size = pairs.len().div_ceil(threads);
            std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(threads);
                for chunk in pairs.chunks(chunk_size) {
                    handles.push(s.spawn(move || pair_transitions(chunk, emission, n_aksharas)));
                }
                for h in handles {
                    let (out, out2) = h.join().expect("pair e-step panicked");
                    for (prev, cur, a1, v) in out {
                        if v >= 1e-4 {
                            *bi.entry((prev, cur)).or_insert(0.0) += v;
                            *bi_ak.entry((a1, cur)).or_insert(0.0) += v;
                        }
                    }
                    for (ctx2, cur, v) in out2 {
                        if v >= 1e-4 {
                            *tri.entry((ctx2, cur)).or_insert(0.0) += v;
                        }
                    }
                }
            });
        } else {
            let (out, out2) = pair_transitions(pairs, emission, n_aksharas);
            for (prev, cur, a1, v) in out {
                if v >= 1e-4 {
                    *bi.entry((prev, cur)).or_insert(0.0) += v;
                    *bi_ak.entry((a1, cur)).or_insert(0.0) += v;
                }
            }
            for (ctx2, cur, v) in out2 {
                if v >= 1e-4 {
                    *tri.entry((ctx2, cur)).or_insert(0.0) += v;
                }
            }
        }
        eprintln!(
            "  [v2] {} transitions, {} ak-level, {} tri",
            bi.len(),
            bi_ak.len(),
            tri.len()
        );

        // Debug: inspect counts for known transitions (na->ma->ste).
        if std::env::var("AKSHAR_V2_DEBUG").is_ok() {
            let id = |s: &str| self.akshara_map.get(s).copied();
            let pack = |c: &str| crate::core::translit_model::pack_chunk_bytes(c.as_bytes());
            for (a1s, c1s, a2s, c2s) in [
                ("न", "na", "म", "ma"),
                ("म", "ma", "स्ते", "ste"),
                ("ना", "na", "मा", "ma"),
            ] {
                if let (Some(a1), Some(a2)) = (id(a1s), id(a2s)) {
                    let prev = pair_key(a1, pack(c1s));
                    let cur = pair_key(a2, pack(c2s));
                    eprintln!(
                        "[dbg] {a1s}+{c1s} -> {a2s}+{c2s}: c={:?} C_prev={:?} ak_c={:?} ak_C={:?}",
                        bi.get(&(prev, cur)),
                        bi.iter().filter(|((p, _), _)| *p == prev)
                            .map(|(_, v)| v).sum::<f64>(),
                        bi_ak.get(&(a1, cur)),
                        bi_ak.iter().filter(|((a, _), _)| *a == a1)
                            .map(|(_, v)| v).sum::<f64>(),
                    );
                } else {
                    eprintln!("[dbg] akshara missing for {a1s}/{a2s}");
                }
            }
        }

        // --- M-step: P(cur | prev) = (c + alpha*P_uni(cur)) / (C_prev + alpha)
        // with the backoff target P_uni = P(a) * P(s | a) (JOINT pair unigram;
        // per-akshara normalisation alone makes rare aksharas with peaked
        // emissions nearly free and junk paths win).
        let mut emit_w: HashMap<u64, f32> = HashMap::with_capacity(self.emission.len() * 8);
        for (a, em) in self.emission.iter().enumerate() {
            for (&chunk, &p) in em {
                if p > 0.0 {
                    emit_w.insert(pair_key(a as u32, chunk), -p.ln() as f32);
                }
            }
        }
        let total_aksharas: u64 = self.unigram_counts.iter().sum();
        let n_ak = self.akshara_list.len() as f64;
        let akshara_uni: Vec<f32> = self
            .unigram_counts
            .iter()
            .map(|&c| {
                let p = (c as f64 + 0.5) / (total_aksharas as f64 + 0.5 * n_ak);
                -p.ln() as f32
            })
            .collect();
        let uni_logprob = |k: u64| -> f64 {
            let emit = emit_w.get(&k).copied().unwrap_or(12.0);
            let a_uni = akshara_uni
                .get(pair_akshara(k) as usize)
                .copied()
                .unwrap_or(14.0);
            emit as f64 + a_uni as f64
        };

        // --- Kneser-Ney-style smoothed backoff hierarchy:
        //   P(cur | prev_pair) = max(c-d,0)/C + lam1 * P(cur | prev_akshara)
        //   P(cur | prev_akshara) = max(c-d,0)/C + lam2 * P_uni(cur)
        // Additive smoothing would make single observations in rare contexts
        // near-certain; the absolute discount keeps them honest.
        let d = config.kn_discount;

        // Level-2 stats: per prev-akshara totals/distincts over cur pairs.
        let mut ak_totals: HashMap<u32, f64> = HashMap::new();
        let mut ak_distinct: HashMap<u32, u64> = HashMap::new();
        for (&(a1, cur), &c) in &bi_ak {
            *ak_totals.entry(a1).or_insert(0.0) += c;
            *ak_distinct.entry(a1).or_insert(0) += 1;
        }
        let lower_ak = |cur: u64| -> f64 { uni_logprob(cur) };
        let p_ak = |a1: u32, cur: u64| -> f64 {
            let counts = &bi_ak;
            let c = counts.get(&(a1, cur)).copied().unwrap_or(0.0);
            let c_total = ak_totals.get(&a1).copied().unwrap_or(0.0);
            let lam = if c_total > 0.0 {
                (d * ak_distinct.get(&a1).copied().unwrap_or(0) as f64) / c_total
            } else {
                1.0
            };
            (c - d).max(0.0) / c_total.max(1e-12) + lam * lower_ak(cur)
        };

        // Level-1: per prev-pair totals/distincts.
        let mut bi_totals: HashMap<u64, f64> = HashMap::new();
        let mut bi_distinct: HashMap<u64, u64> = HashMap::new();
        for (&(prev, _), &c) in &bi {
            *bi_totals.entry(prev).or_insert(0.0) += c;
            *bi_distinct.entry(prev).or_insert(0) += 1;
        }
        let mut bi_out: HashMap<(u64, u64), f32> = HashMap::with_capacity(bi.len());
        for (&(prev, cur), _) in &bi {
            let c = bi.get(&(prev, cur)).copied().unwrap_or(0.0);
            let c_total = bi_totals.get(&prev).copied().unwrap_or(0.0);
            let lam = if c_total > 0.0 {
                (d * bi_distinct.get(&prev).copied().unwrap_or(0) as f64) / c_total
            } else {
                1.0
            };
            let p = (c - d).max(0.0) / c_total.max(1e-12)
                + lam * p_ak(pair_akshara(prev), cur);
            if p > 1e-12 {
                bi_out.insert((prev, cur), -p.ln() as f32);
            }
        }
        let mut bi_ak_out: HashMap<(u32, u64), f32> = HashMap::with_capacity(bi_ak.len());
        for (&(a1, cur), _) in &bi_ak {
            let p = p_ak(a1, cur);
            if p > 1e-12 {
                bi_ak_out.insert((a1, cur), -p.ln() as f32);
            }
        }

        // Depth-2 level: P(cur | ctx2) with the pair-bigram as lower order.
        let p_bi = |prev: u64, cur: u64| -> f64 {
            bi_out
                .get(&(prev, cur))
                .map(|&w| (-w as f64).exp())
                .unwrap_or_else(|| {
                    // Hard fallback mirrors the decoder's chain.
                    p_ak(pair_akshara(prev), cur)
                })
        };
        let mut tri_totals: HashMap<(u64, u64), f64> = HashMap::new();
        let mut tri_distinct: HashMap<(u64, u64), u64> = HashMap::new();
        for (&(ctx2, _), &c) in &tri {
            *tri_totals.entry(ctx2).or_insert(0.0) += c;
            *tri_distinct.entry(ctx2).or_insert(0) += 1;
        }
        let mut tri_out: HashMap<((u64, u64), u64), f32> = HashMap::with_capacity(tri.len());
        for (&(ctx2, cur), _) in &tri {
            let c = tri.get(&(ctx2, cur)).copied().unwrap_or(0.0);
            let c_total = tri_totals.get(&ctx2).copied().unwrap_or(0.0);
            let lam = if c_total > 0.0 {
                (d * tri_distinct.get(&ctx2).copied().unwrap_or(0) as f64) / c_total
            } else {
                1.0
            };
            let p = (c - d).max(0.0) / c_total.max(1e-12) + lam * p_bi(ctx2.1, cur);
            if p > 1e-12 {
                tri_out.insert((ctx2, cur), -p.ln() as f32);
            }
        }

        // Word-start prior from the same corpus counts as the v1 LM.
        let word_start = self
            .word_initial
            .iter()
            .map(|&c| {
                let p = (c as f64 + 0.5) / (self.total_words as f64 + 0.5 * n_ak);
                -p.ln() as f32
            })
            .collect();

        // Dense akshara KN LM (v1's tables) — the backbone the pair grammar
        // refines.  build_kn_lm writes into a throwaway TranslitModel.
        let mut lm_holder = TranslitModel {
            aksharas: self.akshara_list.clone(),
            ..TranslitModel::default()
        };
        self.build_kn_lm(&mut lm_holder, config.kn_discount);
        let mut ak_lm = AkLm {
            bigrams: std::mem::take(&mut lm_holder.bigrams),
            backoff: std::mem::take(&mut lm_holder.backoff),
            unigram_kn: std::mem::take(&mut lm_holder.unigram_kn),
            word_start: std::mem::take(&mut lm_holder.word_start),
            trigram_keys: std::mem::take(&mut lm_holder.trigram_keys),
            trigrams: std::mem::take(&mut lm_holder.trigrams),
            trigram_backoff: std::mem::take(&mut lm_holder.trigram_backoff),
            ..AkLm::default()
        };
        ak_lm.build_index();

        eprintln!(
            "  [v2] {} emit pairs, {} bi, {} bi_ak, {} tri",
            emit_w.len(),
            bi_out.len(),
            bi_ak_out.len(),
            tri_out.len()
        );
        PairModel {
            aksharas: self.akshara_list.clone(),
            akshara_uni,
            emit_w,
            bi: bi_out,
            bi_ak: bi_ak_out,
            tri: tri_out,
            word_start,
            ak_lm,
        }
    }
}

/// A bare-consonant akshara carries an inherent schwa that roman spellings may
/// write as a trailing "a" or drop.  Such aksharas end in a consonant or halanta
/// (never a matra / anusvara / visarga).
fn is_bare_consonant(akshara: &str) -> bool {
    let Some(last) = akshara.chars().last() else {
        return false;
    };
    let cp = last as u32;
    (0x0915..=0x0939).contains(&cp)
        || (0x0958..=0x095F).contains(&cp)
        || matches!(
            cp,
            0x0931 | 0x0934 | 0x0978 | 0x0979 | 0x097A | 0x097B | 0x097D | 0x094D
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_consonant_detection() {
        assert!(is_bare_consonant("क"));
        assert!(is_bare_consonant("च"));
        assert!(is_bare_consonant("क्"));
        assert!(!is_bare_consonant("का"));
        assert!(!is_bare_consonant("कि"));
        assert!(!is_bare_consonant("कं"));
        assert!(!is_bare_consonant("अ"));
    }

    #[test]
    fn trainer_interns_aksharas() {
        let mut t = Trainer::new();
        t.add_pair("ka", "क");
        t.add_pair("kha", "ख");
        assert_eq!(t.akshara_list, vec!["क", "ख"]);
        assert_eq!(t.ingested, 2);
    }

    #[test]
    fn trainer_counts_bigrams() {
        let mut t = Trainer::new();
        t.add_pair("ka", "क");
        t.add_pair("kama", "कम");
        assert_eq!(t.unigram_counts, vec![2, 1]);
        assert_eq!(t.bigram_counts.get(&(0, 1)), Some(&1));
        assert_eq!(t.continuation[1], 1);
    }

    #[test]
    fn trainer_skips_dirty_roman() {
        let mut t = Trainer::new();
        t.add_pair("Ka", "क");
        assert_eq!(t.ingested, 0);
        assert_eq!(t.unigram_counts[0], 1);
    }

    #[test]
    fn finalize_produces_valid_model() {
        let mut t = Trainer::new();
        t.add_pair("ka", "क");
        t.add_pair("kama", "कम");
        t.add_pair("nepal", "नेपाल");
        t.add_pair("namaste", "नमस्ते");
        let model = t.finalize(&TrainerConfig::default());
        assert!(model.validate());
        assert!(model.aksharas.len() >= 4);
        let kid = model.akshara_id("क").unwrap();
        let top = model.top_emissions(kid, 3);
        assert!(!top.is_empty());
        assert!(top.iter().any(|(c, _)| c == "ka"));
    }

    #[test]
    fn em_learns_expected_emission() {
        let mut t = Trainer::new();
        // म is written "ma" 3x and "m" once; EM should learn P(ma|म) > P(m|म).
        for _ in 0..3 {
            t.add_pair("nama", "नम");
        }
        t.add_pair("namaste", "नमस्ते");
        let model = t.finalize(&TrainerConfig {
            iterations: 8,
            ..Default::default()
        });
        let nid = model.akshara_id("न").unwrap();
        let p_na = model.emission_prob(nid, "na");
        let p_n = model.emission_prob(nid, "n");
        assert!(p_na > p_n, "P(na|न) should exceed P(n|न): {p_na} vs {p_n}");
        let mid = model.akshara_id("म").unwrap();
        let p_ma = model.emission_prob(mid, "ma");
        let p_m = model.emission_prob(mid, "m");
        assert!(p_ma > p_m, "P(ma|म) should exceed P(m|म): {p_ma} vs {p_m}");
    }

    #[test]
    fn pack_helpers_agree() {
        assert_eq!(
            pack_chunk_bytes(b"ka"),
            crate::core::translit_model::pack_chunk("ka")
        );
        assert_eq!(pack_chunk_bytes(b""), 0);
    }
}
