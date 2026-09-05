// File: src/core/crf.rs
//
// S3: conditional random field over the akshara lattice.
//
// Replaces the hand-composed `Σ emission + λ·Σ LM` path score with weights
// trained jointly by maximizing the conditional likelihood of the gold word
// given the roman string:
//
//     P(gold | roman) = Z_gold / Z_all      (log-likelihood is always ≤ 0)
//
// Three feature templates (the only place weights live):
//
//     emit[(a, c)]  — akshara a spelled as chunk id c
//     trans[(p, a)] — akshara bigram p -> a
//     start[a]      — word-initial akshara
//
// Correctness invariants (the bugs the first implementation had):
//
//   1. ONE measure: both lattices score edges through the same
//      `edge_weight()` — gold edges include their transition/start parts, so
//      Z_gold ≤ Z_all holds and the log-likelihood can never be positive.
//   2. ONE gradient: w += lr * (gold_count − total_count − l2 * w). The
//      expected-count term must never be applied twice (a double subtraction
//      drives every trained weight negative and unseen zero-weight paths win
//      at decode — the failure that collapsed the first attempt).
//   3. L2 is applied lazily, only to feature keys touched by the current
//      pair (full-map sweeps are infeasible at 15M+ features).
//
// Numerics: everything in log space with max-shifted log-sum-exp; posteriors
// below 1e-12 are dropped.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const MAX_CHUNK: usize = 5;
/// Akshara candidates per (position, chunk) taken from the emission index;
/// gold aksharas are always unioned in on top.
pub const TOP_N_PER_CHUNK: usize = 10;
/// Decode-time penalties for feature keys never seen in training. Unseen
/// keys must be expensive, never free (free unseen edges are exactly how the
/// first attempt's decode collapsed).
pub const UNSEEN_EMIT: f64 = 8.0;
pub const UNSEEN_TRANS: f64 = 6.0;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[allow(clippy::type_complexity)]
pub struct CrfModel {
    /// (akshara id, chunk id) -> weight, higher = more likely.
    pub emit: HashMap<(u32, u32), f64>,
    /// (previous akshara id, akshara id) -> weight.
    pub trans: HashMap<(u32, u32), f64>,
    /// Word-initial akshara id -> weight.
    pub start: HashMap<u32, f64>,
    /// Chunk string -> id (feature key space, shared with `emit`).
    pub chunk_ids: HashMap<String, u32>,
    /// Decoding dictionary: chunk string -> trained aksharas, sorted by emit
    /// weight descending. Built from the trained `emit` keys after training —
    /// every decode edge therefore carries a real trained weight; free edges
    /// are impossible by construction.
    pub decode_cands: HashMap<String, Vec<u32>>,
}

impl CrfModel {
    pub fn chunk_id(&self, chunk: &str) -> Option<u32> {
        self.chunk_ids.get(chunk).copied()
    }

    /// Decode-time edge parts with base generative model scores.
    /// Returns (adjusted_emit, adjusted_lm).
    pub fn edge_parts_with_base(
        &self,
        prev: Option<u32>,
        a: u32,
        c: u32,
        base_emit: f64,
        base_lm: f64,
    ) -> (f64, f64) {
        let e = (base_emit - self.emit.get(&(a, c)).copied().unwrap_or(0.0)).max(0.0);
        let t = match prev {
            None => (base_lm - self.start.get(&a).copied().unwrap_or(0.0)).max(0.0),
            Some(p) => (base_lm - self.trans.get(&(p, a)).copied().unwrap_or(0.0)).max(0.0),
        };
        (e, t)
    }

    /// Decode-time edge parts (emission-part, transition-part) as negative
    /// log weights (lower = better). The split preserves the reranker's
    /// emit/lm feature semantics.
    pub fn edge_parts(&self, prev: Option<u32>, a: u32, c: u32) -> (f64, f64) {
        let e = -self.emit.get(&(a, c)).copied().unwrap_or(-UNSEEN_EMIT);
        let t = match prev {
            None => -self.start.get(&a).copied().unwrap_or(-UNSEEN_TRANS),
            Some(p) => -self.trans.get(&(p, a)).copied().unwrap_or(-UNSEEN_TRANS),
        };
        (e, t)
    }
}

#[inline]
pub fn logsumexp2(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let (max, min) = if a > b { (a, b) } else { (b, a) };
    let diff = min - max;
    if diff < -36.0 {
        max
    } else {
        max + diff.exp().ln_1p()
    }
}

#[inline]
pub fn logsumexp(xs: &[f64]) -> f64 {
    xs.iter().copied().fold(f64::NEG_INFINITY, logsumexp2)
}

#[inline]
fn lse_into(slot: &mut f64, v: f64) {
    *slot = logsumexp2(*slot, v);
}

/// One pair's (or one merged minibatch's) gradient: per-feature-key
/// (gold count − total count). Weights are updated by the trainer's apply
/// step, so `pair_gradient` is a pure function of the current weights and
/// minibatches can be computed in parallel against an immutable snapshot.
#[derive(Default)]
pub struct Grad {
    pub emit: HashMap<(u32, u32), f64>,
    pub trans: HashMap<(u32, u32), f64>,
    pub start: HashMap<u32, f64>,
}

impl Grad {
    pub fn merge(&mut self, other: Grad) {
        for (k, v) in other.emit {
            *self.emit.entry(k).or_insert(0.0) += v;
        }
        for (k, v) in other.trans {
            *self.trans.entry(k).or_insert(0.0) += v;
        }
        for (k, v) in other.start {
            *self.start.entry(k).or_insert(0.0) += v;
        }
    }
}

/// Trainer: minibatch-parallel SGD on the conditional log-likelihood.
/// Gradients are computed in parallel against an immutable weight snapshot
/// (`pair_gradient` is pure) and applied sequentially (`apply_gradient`) —
/// deterministic, race-free, and exactly the objective of sequential SGD
/// with the same minibatch partition.
pub struct CrfTrainer {
    pub model: CrfModel,
    pub lr: f64,
    pub l2: f64,
}

impl CrfTrainer {
    pub fn new(lr: f64, l2: f64) -> Self {
        Self {
            model: CrfModel::default(),
            lr,
            l2,
        }
    }

    /// Registers a chunk string in the id space (called during ingestion).
    pub fn chunk_id(&mut self, chunk: &str) -> u32 {
        let next = self.model.chunk_ids.len() as u32;
        *self.model.chunk_ids.entry(chunk.to_string()).or_insert(next)
    }
    #[inline]
    fn edge_weight_with_base(&self, prev: Option<u32>, a: u32, c: u32, base: f64) -> f64 {
        let mut w = base;
        w += self.model.emit.get(&(a, c)).copied().unwrap_or(0.0);
        w += match prev {
            None => self.model.start.get(&a).copied().unwrap_or(0.0),
            Some(p) => self.model.trans.get(&(p, a)).copied().unwrap_or(0.0),
        };
        w
    }

    /// Pure gradient computation for one pair: returns the conditional
    /// log-likelihood log P(gold | roman) (always ≤ 0 up to float slack)
    /// and the per-key gradient (gold counts − total counts). No mutation.
    pub fn pair_gradient(
        &self,
        roman: &str,
        gold: &[u32],
        candidates: &dyn Fn(&str) -> Vec<u32>,
    ) -> Option<(f64, Grad)> {
        self.pair_gradient_with_base(roman, gold, candidates, None)
    }

    #[allow(clippy::type_complexity)]
    pub fn pair_gradient_with_base(
        &self,
        roman: &str,
        gold: &[u32],
        candidates: &dyn Fn(&str) -> Vec<u32>,
        base_fn: Option<&dyn Fn(Option<u32>, u32, &str) -> f64>,
    ) -> Option<(f64, Grad)> {
        let m = roman.len();
        let n = gold.len();
        if m == 0 || n == 0 || n > m || m > n * MAX_CHUNK {
            return None;
        }

        // Chunk enumeration per start position: (length, chunk id). All
        // chunk ids must already be registered (ingestion-time); unknown
        // chunks return None from the id map and the pair is skipped.
        let mut chunks: Vec<Vec<(usize, u32)>> = vec![Vec::new(); m];
        for i in 0..m {
            for l in 1..=MAX_CHUNK.min(m - i) {
                let cid = self.model.chunk_id(&roman[i..i + l])?;
                chunks[i].push((l, cid));
            }
        }

        // ---------------- gold pass ----------------
        // fg[j][i]: log-mass of alignments of the first j gold aksharas that
        // consume exactly i characters. Push form over chunk start positions.
        let mut fg = vec![vec![f64::NEG_INFINITY; m + 1]; n + 1];
        let mut bg = vec![vec![f64::NEG_INFINITY; m + 1]; n + 1];
        fg[0][0] = 0.0;
        for j in 0..n {
            for i in 0..m {
                let cur = fg[j][i];
                if cur == f64::NEG_INFINITY {
                    continue;
                }
                let prev = if j == 0 { None } else { Some(gold[j - 1]) };
                for &(l, cid) in &chunks[i] {
                    if i + l > m {
                        continue;
                    }
                    let chunk_str = &roman[i..i + l];
                    let base_w = base_fn.map_or(0.0, |b| b(prev, gold[j], chunk_str));
                    let w = self.edge_weight_with_base(prev, gold[j], cid, base_w);
                    lse_into(&mut fg[j + 1][i + l], cur + w);
                }
            }
        }
        let z_gold = fg[n][m];
        if !z_gold.is_finite() {
            return None;
        }

        bg[n][m] = 0.0;
        for j in (0..n).rev() {
            for i in 0..m {
                let prev = if j == 0 { None } else { Some(gold[j - 1]) };
                let mut val = f64::NEG_INFINITY;
                for &(l, cid) in &chunks[i] {
                    if i + l > m {
                        continue;
                    }
                    let bv = bg[j + 1][i + l];
                    if bv == f64::NEG_INFINITY {
                        continue;
                    }
                    let chunk_str = &roman[i..i + l];
                    let base_w = base_fn.map_or(0.0, |b| b(prev, gold[j], chunk_str));
                    let w = self.edge_weight_with_base(prev, gold[j], cid, base_w);
                    lse_into(&mut val, bv + w);
                }
                bg[j][i] = val;
            }
        }
        debug_assert!(
            (bg[0][0] - z_gold).abs() < 1e-5,
            "gold backward/forward partition mismatch: bg[0][0]={}, fg[n][m]={}",
            bg[0][0], z_gold
        );

        // Gold-expected feature counts and record actually reachable gold aksharas per (i, li).
        let mut g_emit: HashMap<(u32, u32), f64> = HashMap::new();
        let mut g_trans: HashMap<(u32, u32), f64> = HashMap::new();
        let mut g_start: HashMap<u32, f64> = HashMap::new();
        let mut gold_used: Vec<Vec<Vec<u32>>> = (0..m).map(|i| vec![Vec::new(); chunks[i].len()]).collect();

        for j in 0..n {
            let prev = if j == 0 { None } else { Some(gold[j - 1]) };
            for i in 0..m {
                let cur = fg[j][i];
                if cur == f64::NEG_INFINITY {
                    continue;
                }
                for (li, &(l, cid)) in chunks[i].iter().enumerate() {
                    if i + l > m {
                        continue;
                    }
                    let bv = bg[j + 1][i + l];
                    if bv == f64::NEG_INFINITY {
                        continue;
                    }
                    let chunk_str = &roman[i..i + l];
                    let base_w = base_fn.map_or(0.0, |b| b(prev, gold[j], chunk_str));
                    let w = self.edge_weight_with_base(prev, gold[j], cid, base_w);
                    let log_p = cur + w + bv - z_gold;
                    if log_p < -27.63 { // threshold 1e-12 in log-space
                        continue;
                    }
                    let post = log_p.exp();
                    *g_emit.entry((gold[j], cid)).or_insert(0.0) += post;
                    match prev {
                        None => *g_start.entry(gold[0]).or_insert(0.0) += post,
                        Some(p) => *g_trans.entry((p, gold[j])).or_insert(0.0) += post,
                    }
                    if !gold_used[i][li].contains(&gold[j]) {
                        gold_used[i][li].push(gold[j]);
                    }
                }
            }
        }

        // Candidate aksharas per (pos, length index): emission-index top-N,
        // unioned with the gold aksharas that are actually reachable at this (pos, length).
        let mut cand: Vec<Vec<Vec<u32>>> = vec![Vec::new(); m];
        for i in 0..m {
            cand[i] = vec![Vec::new(); chunks[i].len()];
            for (li, &(l, _)) in chunks[i].iter().enumerate() {
                let mut list: Vec<u32> = candidates(&roman[i..i + l])
                    .into_iter()
                    .take(TOP_N_PER_CHUNK)
                    .collect();
                for &g in &gold_used[i][li] {
                    if !list.contains(&g) {
                        list.push(g);
                    }
                }
                list.sort_unstable();
                list.dedup();
                cand[i][li] = list;
            }
        }

        // ---------------- total pass ----------------
        // States are (position, previous akshara); `None` = word start.
        let mut alpha: Vec<HashMap<Option<u32>, f64>> = vec![HashMap::new(); m + 1];
        alpha[0].insert(None, 0.0);
        for i in 0..m {
            if alpha[i].is_empty() {
                continue;
            }
            let alpha_i: Vec<(Option<u32>, f64)> =
                alpha[i].iter().map(|(k, &v)| (*k, v)).collect();
            for &(l, cid) in &chunks[i] {
                if i + l > m {
                    continue;
                }
                let chunk_str = &roman[i..i + l];
                for &a in &cand[i][l - 1] {
                    for &(p, av) in alpha_i.iter() {
                        let base_w = base_fn.map_or(0.0, |b| b(p, a, chunk_str));
                        let w = self.edge_weight_with_base(p, a, cid, base_w);
                        lse_into(alpha[i + l].entry(Some(a)).or_insert(f64::NEG_INFINITY),
                                 av + w);
                    }
                }
            }
        }
        let z_all = {
            let finals: Vec<f64> = alpha[m].values().copied().collect();
            if finals.is_empty() {
                f64::NEG_INFINITY
            } else {
                logsumexp(&finals)
            }
        };
        if !z_all.is_finite() {
            return None;
        }
        debug_assert!(
            z_gold <= z_all + 1e-6,
            "measure broken: z_gold {z_gold} > z_all {z_all}"
        );

        let mut beta: Vec<HashMap<Option<u32>, f64>> = vec![HashMap::new(); m + 1];
        for &k in alpha[m].keys() {
            beta[m].insert(k, 0.0);
        }
        for i in (0..m).rev() {
            if alpha[i].is_empty() {
                continue;
            }
            let mut bi: HashMap<Option<u32>, f64> = HashMap::new();
            for &(l, cid) in &chunks[i] {
                if i + l > m {
                    continue;
                }
                let chunk_str = &roman[i..i + l];
                for &a in &cand[i][l - 1] {
                    if let Some(&bv) = beta[i + l].get(&Some(a)) {
                        for p in alpha[i].keys() {
                            let base_w = base_fn.map_or(0.0, |b| b(*p, a, chunk_str));
                            let w = self.edge_weight_with_base(*p, a, cid, base_w);
                            lse_into(bi.entry(*p).or_insert(f64::NEG_INFINITY), bv + w);
                        }
                    }
                }
            }
            beta[i] = bi;
        }

        // Total-expected feature counts.
        let mut a_emit: HashMap<(u32, u32), f64> = HashMap::new();
        let mut a_trans: HashMap<(u32, u32), f64> = HashMap::new();
        let mut a_start: HashMap<u32, f64> = HashMap::new();
        for i in 0..m {
            if alpha[i].is_empty() {
                continue;
            }
            let alpha_i: Vec<(Option<u32>, f64)> =
                alpha[i].iter().map(|(k, &v)| (*k, v)).collect();
            for &(l, cid) in &chunks[i] {
                if i + l > m {
                    continue;
                }
                let chunk_str = &roman[i..i + l];
                for &a in &cand[i][l - 1] {
                    let bv = match beta[i + l].get(&Some(a)) {
                        Some(&v) if v.is_finite() => v,
                        _ => continue,
                    };
                    for &(p, av) in alpha_i.iter() {
                        let base_w = base_fn.map_or(0.0, |b| b(p, a, chunk_str));
                        let w = self.edge_weight_with_base(p, a, cid, base_w);
                        let post = (av + w + bv - z_all).exp();
                        if post < 1e-12 {
                            continue;
                        }
                        *a_emit.entry((a, cid)).or_insert(0.0) += post;
                        match p {
                            None => *a_start.entry(a).or_insert(0.0) += post,
                            Some(pk) => *a_trans.entry((pk, a)).or_insert(0.0) += post,
                        }
                    }
                }
            }
        }

        let mut grad = Grad::default();
        for (&k, &gv) in &g_emit {
            let av = a_emit.get(&k).copied().unwrap_or(0.0);
            let d = gv - av;
            if d.abs() > 1e-12 {
                grad.emit.insert(k, d);
            }
        }
        for (&k, &av) in &a_emit {
            if !g_emit.contains_key(&k) && av > 1e-12 {
                grad.emit.insert(k, -av);
            }
        }
        for (&k, &gv) in &g_trans {
            let av = a_trans.get(&k).copied().unwrap_or(0.0);
            let d = gv - av;
            if d.abs() > 1e-12 {
                grad.trans.insert(k, d);
            }
        }
        for (&k, &av) in &a_trans {
            if !g_trans.contains_key(&k) && av > 1e-12 {
                grad.trans.insert(k, -av);
            }
        }
        for (&k, &gv) in &g_start {
            let av = a_start.get(&k).copied().unwrap_or(0.0);
            let d = gv - av;
            if d.abs() > 1e-12 {
                grad.start.insert(k, d);
            }
        }
        for (&k, &av) in &a_start {
            if !g_start.contains_key(&k) && av > 1e-12 {
                grad.start.insert(k, -av);
            }
        }
        Some((z_gold - z_all, grad))
    }

    /// Apply a merged minibatch gradient (SUM over pairs). Sum-scaling —
    /// not batch-mean — keeps each feature's per-occurrence update magnitude
    /// identical to sequential SGD: our features are sparse (a key appears
    /// in a handful of pairs per minibatch), so mean-scaling would shrink
    /// every update by the batch size and stall learning.
    pub fn apply_gradient(&mut self, grad: &Grad) {
        let scale = self.lr;
        for (k, g) in &grad.emit {
            let e = self.model.emit.entry(*k).or_insert(0.0);
            *e += scale * g - self.lr * self.l2 * *e;
        }
        for (k, g) in &grad.trans {
            let e = self.model.trans.entry(*k).or_insert(0.0);
            *e += scale * g - self.lr * self.l2 * *e;
        }
        for (k, g) in &grad.start {
            let e = self.model.start.entry(*k).or_insert(0.0);
            *e += scale * g - self.lr * self.l2 * *e;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_finite_difference() {
        let mut trainer = CrfTrainer::new(0.1, 0.0);
        let roman = "namaste";
        let gold = vec![0, 1, 3];

        let m = roman.len();
        for i in 0..m {
            for l in 1..=MAX_CHUNK.min(m - i) {
                trainer.chunk_id(&roman[i..i + l]);
            }
        }
        let c_na = trainer.model.chunk_id("na").unwrap();
        let c_ma = trainer.model.chunk_id("ma").unwrap();
        let c_ste = trainer.model.chunk_id("ste").unwrap();

        // 0: na (gold), 1: ma (gold), 2: naa (competing), 3: ste (gold)
        trainer.model.emit.insert((0, c_na), 0.5);
        trainer.model.emit.insert((2, c_na), 0.8);
        trainer.model.emit.insert((1, c_ma), 0.3);
        trainer.model.emit.insert((3, c_ste), 0.6);

        trainer.model.start.insert(0, 0.2);
        trainer.model.start.insert(2, 0.4);

        trainer.model.trans.insert((0, 1), 0.1);
        trainer.model.trans.insert((2, 1), 0.5);
        trainer.model.trans.insert((1, 3), 0.2);

        let cand_fn = |chunk: &str| -> Vec<u32> {
            match chunk {
                "na" => vec![0, 2],
                "ma" => vec![1],
                "ste" => vec![3],
                _ => vec![],
            }
        };

        let (ll, grad) = trainer.pair_gradient(roman, &gold, &cand_fn).expect("gradient");
        assert!(ll.is_finite() && ll <= 0.0);

        let compute_ll = |t: &CrfTrainer| -> f64 {
            t.pair_gradient(roman, &gold, &cand_fn).unwrap().0
        };

        let eps = 1e-5;

        // Check emit gradients
        for &(a, c) in &[(0, c_na), (2, c_na), (1, c_ma), (3, c_ste)] {
            let orig = trainer.model.emit.get(&(a, c)).copied().unwrap_or(0.0);

            trainer.model.emit.insert((a, c), orig + eps);
            let ll_plus = compute_ll(&trainer);

            trainer.model.emit.insert((a, c), orig - eps);
            let ll_minus = compute_ll(&trainer);

            trainer.model.emit.insert((a, c), orig);

            let num_grad = (ll_plus - ll_minus) / (2.0 * eps);
            let ana_grad = grad.emit.get(&(a, c)).copied().unwrap_or(0.0);

            assert!(
                (num_grad - ana_grad).abs() < 1e-4,
                "emit ({a}, {c}) grad mismatch: num={num_grad}, ana={ana_grad}"
            );
        }

        // Check start gradients
        for &a in &[0, 2] {
            let orig = trainer.model.start.get(&a).copied().unwrap_or(0.0);

            trainer.model.start.insert(a, orig + eps);
            let ll_plus = compute_ll(&trainer);

            trainer.model.start.insert(a, orig - eps);
            let ll_minus = compute_ll(&trainer);

            trainer.model.start.insert(a, orig);

            let num_grad = (ll_plus - ll_minus) / (2.0 * eps);
            let ana_grad = grad.start.get(&a).copied().unwrap_or(0.0);

            assert!(
                (num_grad - ana_grad).abs() < 1e-4,
                "start {a} grad mismatch: num={num_grad}, ana={ana_grad}"
            );
        }

        // Check trans gradients
        for &(p, a) in &[(0, 1), (2, 1), (1, 3)] {
            let orig = trainer.model.trans.get(&(p, a)).copied().unwrap_or(0.0);

            trainer.model.trans.insert((p, a), orig + eps);
            let ll_plus = compute_ll(&trainer);

            trainer.model.trans.insert((p, a), orig - eps);
            let ll_minus = compute_ll(&trainer);

            trainer.model.trans.insert((p, a), orig);

            let num_grad = (ll_plus - ll_minus) / (2.0 * eps);
            let ana_grad = grad.trans.get(&(p, a)).copied().unwrap_or(0.0);

            assert!(
                (num_grad - ana_grad).abs() < 1e-4,
                "trans ({p}, {a}) grad mismatch: num={num_grad}, ana={ana_grad}"
            );
        }
    }

    #[test]
    fn test_toy_training_and_partition_consistency() {
        let mut trainer = CrfTrainer::new(0.5, 1e-4);

        // Vocabulary:
        // 0: "ka", 1: "la", 2: "ma"
        // Words:
        // 1. "kal" -> [0, 1] (ka, la)
        // 2. "kam" -> [0, 2] (ka, ma)
        // 3. "mal" -> [2, 1] (ma, la)
        let pairs = vec![
            ("kal", vec![0, 1]),
            ("kam", vec![0, 2]),
            ("mal", vec![2, 1]),
        ];

        for (roman, _) in &pairs {
            let m = roman.len();
            for i in 0..m {
                for l in 1..=MAX_CHUNK.min(m - i) {
                    trainer.chunk_id(&roman[i..i + l]);
                }
            }
        }

        let cand_fn = |chunk: &str| -> Vec<u32> {
            match chunk {
                "k" | "ka" => vec![0, 3], // 0 is gold, 3 is competing
                "l" | "la" => vec![1],
                "m" | "ma" => vec![2],
                _ => vec![],
            }
        };

        // Train for 20 epochs
        let mut prev_ll = f64::NEG_INFINITY;
        for _epoch in 0..20 {
            let mut epoch_ll = 0.0;
            for (roman, gold) in &pairs {
                let (ll, grad) = trainer.pair_gradient(roman, gold, &cand_fn).expect("pair grad");
                epoch_ll += ll;
                trainer.apply_gradient(&grad);
            }
            if prev_ll.is_finite() {
                // Log-likelihood should increase or be close
                assert!(epoch_ll >= prev_ll - 0.05, "LL decreased: {} < {}", epoch_ll, prev_ll);
            }
            prev_ll = epoch_ll;
        }

        eprintln!("model emit: {:?}", trainer.model.emit);
        eprintln!("model trans: {:?}", trainer.model.trans);
        eprintln!("model start: {:?}", trainer.model.start);

        // After training, log-likelihood should be very close to 0 (high probability on gold)
        assert!(prev_ll > -0.5, "Expected converged LL close to 0, got {}", prev_ll);

        // Test Viterbi score on "kal": gold [0, 1] vs competing [3, 1]
        let c_ka = trainer.model.chunk_id("ka").unwrap();
        let c_l = trainer.model.chunk_id("l").unwrap();
        let (e1, t1) = trainer.model.edge_parts(None, 0, c_ka);
        let (e2, t2) = trainer.model.edge_parts(Some(0), 1, c_l);
        let gold_score = e1 + t1 + e2 + t2;

        let (bad_e1, bad_t1) = trainer.model.edge_parts(None, 3, c_ka);
        let (bad_e2, bad_t2) = trainer.model.edge_parts(Some(3), 1, c_l);
        let bad_score = bad_e1 + bad_t1 + bad_e2 + bad_t2;

        eprintln!("gold score: {}, bad score: {}", gold_score, bad_score);

        // In decoder, lower score is better (cost = -weight)
        assert!(
            gold_score < bad_score,
            "Gold path should have better (lower) score than bad path: gold={}, bad={}",
            gold_score, bad_score
        );
    }
}

