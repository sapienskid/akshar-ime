// File: src/bin/train_reranker.rs
//
// High-throughput, leakage-free discriminative reranker trainer (W1).
//
// Trains a log-linear softmax reranker directly on k-best candidate lists
// with both dense shape features and sparse lexicalized hash templates:
//   - Final/first akshara x roman char
//   - (Matra id x preceding consonant id)
//   - Length delta buckets
//
// Includes an interactive progress bar with ETA, throughput, loss tracking,
// checkpointing for overnight runs, and a rapid smoke-test mode.
//
// Usage:
//   cargo run --release --bin train_reranker -- [options]
//     --train <path>       (default: data/aksharantar/train_devanagari.jsonl)
//     --valid <path>       (default: data/aksharantar/valid_devanagari.jsonl)
//     --test  <path>       (default: data/aksharantar/test_devanagari.jsonl)
//     --model <path>       (default: data/translit_model.bin)
//     --vocab <path>       (default: data/word_freq_text.bin)
//     --out   <path>       (default: src/core/reranker_weights.rs)
//     --smoke <n>          (run fast smoke test on <n> training pairs)
//     --epochs <n>         (default: 6)
//     --batch-size <n>     (default: 256)
//     --lr <f>             (default: 0.05)
//     --l2 <f>             (default: 1e-4)

use akshar_ime::core::akshara;
use akshar_ime::core::decoder::{DecoderConfig, ModelDecoder, DecodedCandidate};
use akshar_ime::core::reranker::{
    extract_dense_features, extract_sparse_features, FreqRanks, DENSE_DIM, HASH_SIZE,
};
use akshar_ime::core::translit_model::TranslitModel;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Deserialize)]
struct Record {
    #[serde(rename = "english word", alias = "english")]
    english: String,
    #[serde(rename = "native word", alias = "native")]
    native: String,
}

#[derive(Clone)]
struct CandFeatures {
    dense: [f64; DENSE_DIM],
    sparse: Vec<usize>,
}

#[derive(Clone)]
#[allow(dead_code)]
struct PairCase {
    roman: String,
    gold: String,
    source: String,
    cands: Vec<String>,
    feats: Vec<CandFeatures>,
    gold_idx: Option<usize>,
}


fn featurize_candidates_direct(
    detailed: &[DecodedCandidate],
    roman: &str,
    gold: &str,
    source: &str,
    freq: &HashMap<String, u32>,
    ranks: &FreqRanks,
) -> Option<PairCase> {
    if detailed.is_empty() {
        return None;
    }

    let mut order: Vec<&DecodedCandidate> = detailed.iter().collect();
    order.sort_by(|a, b| {
        (a.emit + a.lm)
            .total_cmp(&(b.emit + b.lm))
            .then(a.dev.cmp(&b.dev))
    });

    let heur: Vec<f64> = order
        .iter()
        .map(|c| {
            let f = freq.get(&c.dev).copied().unwrap_or(0);
            c.emit + 0.85 * c.lm - 0.75 * (1.0 + f as f64).ln()
        })
        .collect();
    let mut heur_order: Vec<usize> = (0..order.len()).collect();
    heur_order.sort_by(|a, b| {
        heur[*a]
            .total_cmp(&heur[*b])
            .then(order[*a].dev.cmp(&order[*b].dev))
    });
    let mut heur_rank = vec![0usize; order.len()];
    for (r, i) in heur_order.into_iter().enumerate() {
        heur_rank[i] = r;
    }

    let mut cands = Vec::with_capacity(order.len());
    let mut feats = Vec::with_capacity(order.len());
    let mut gold_idx = None;

    for (i, c) in order.iter().enumerate() {
        if c.dev == gold {
            gold_idx = Some(i);
        }
        let dense = extract_dense_features(c, i, heur[i], heur_rank[i], roman, freq, ranks);
        let aks = akshara::segment(&c.dev);
        let sparse = extract_sparse_features(&c.dev, roman, c.akshara_count, &aks);
        cands.push(c.dev.clone());
        feats.push(CandFeatures { dense, sparse });
    }

    Some(PairCase {
        roman: roman.to_string(),
        gold: gold.to_string(),
        source: source.to_string(),
        cands,
        feats,
        gold_idx,
    })
}

fn featurize_candidates(
    decoder: &ModelDecoder,
    roman: &str,
    gold: &str,
    source: &str,
    freq: &HashMap<String, u32>,
    ranks: &FreqRanks,
    wtrie: Option<&akshar_ime::core::wordtrie::WordTrie>,
) -> Option<PairCase> {
    let detailed = decoder.decode_union(roman, 50, wtrie);
    featurize_candidates_direct(&detailed, roman, gold, source, freq, ranks)
}

#[derive(Deserialize)]
struct DumpCandidate(String, f64, f64, usize);

#[derive(Deserialize)]
#[allow(dead_code)]
struct TrieCandidate(String, f64, u32);

#[derive(Deserialize)]
struct DumpRecord {
    roman: String,
    gold: String,
    source: Option<String>,
    cands: Vec<DumpCandidate>,
    trie: Option<Vec<TrieCandidate>>,
}

fn load_dump_cases(
    path: &str,
    freq: &HashMap<String, u32>,
    ranks: &FreqRanks,
    limit: Option<usize>,
) -> Option<Vec<PairCase>> {
    let file = File::open(path).ok()?;
    let mut cases = Vec::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if let Ok(rec) = serde_json::from_str::<DumpRecord>(&line) {
            let mut detailed: Vec<DecodedCandidate> = rec
                .cands
                .into_iter()
                .map(|DumpCandidate(dev, emit, lm, akshara_count)| DecodedCandidate {
                    dev,
                    emit,
                    lm,
                    akshara_count,
                })
                .collect();
            if let Some(trie) = rec.trie {
                let mut seen: std::collections::HashSet<String> =
                    detailed.iter().map(|c| c.dev.clone()).collect();
                for TrieCandidate(dev, score, _) in trie {
                    if seen.insert(dev.clone()) {
                        let aks_cnt = akshara::segment(&dev).len();
                        detailed.push(DecodedCandidate {
                            dev,
                            emit: score * 0.5,
                            lm: score * 0.5,
                            akshara_count: aks_cnt,
                        });
                    }
                }
            }
            if let Some(c) = featurize_candidates_direct(
                &detailed,
                &rec.roman,
                &rec.gold,
                rec.source.as_deref().unwrap_or("AK"),
                freq,
                ranks,
            ) {
                cases.push(c);
                if let Some(lim) = limit {
                    if cases.len() >= lim {
                        break;
                    }
                }
            }
        }
    }
    Some(cases)
}

struct ProgressTracker {
    total: usize,
    completed: AtomicUsize,
    start: Instant,
    last_print: std::sync::Mutex<Instant>,
}

impl ProgressTracker {
    fn new(total: usize) -> Self {
        Self {
            total,
            completed: AtomicUsize::new(0),
            start: Instant::now(),
            last_print: std::sync::Mutex::new(Instant::now()),
        }
    }

    fn inc(&self, n: usize, status: &str) {
        let cur = self.completed.fetch_add(n, Ordering::Relaxed) + n;
        let mut lp = self.last_print.lock().unwrap();
        if lp.elapsed().as_millis() >= 200 || cur == self.total {
            *lp = Instant::now();
            let elapsed = self.start.elapsed().as_secs_f64();
            let pct = (cur as f64 / self.total.max(1) as f64) * 100.0;
            let rate = if elapsed > 0.0 { cur as f64 / elapsed } else { 0.0 };
            let rem_s = if rate > 0.0 { ((self.total - cur) as f64 / rate) as u64 } else { 0 };
            let eta = format!("{:02}:{:02}", rem_s / 60, rem_s % 60);
            let bar_len = 25;
            let filled = ((pct / 100.0) * bar_len as f64).round() as usize;
            let bar: String = (0..bar_len).map(|i| if i < filled { '=' } else if i == filled { '>' } else { '-' }).collect();

            eprint!("\r[{bar}] {pct:5.1}% ({cur}/{}) | {rate:6.0} pairs/s | ETA: {eta} | {status}   ", self.total);
            if cur == self.total {
                eprintln!();
            }
            let _ = std::io::stderr().flush();
        }
    }
}

fn main() {
    let mut train_path = "data/aksharantar/train_devanagari.jsonl".to_string();
    let mut valid_path = "data/aksharantar/valid_devanagari.jsonl".to_string();
    let mut test_path = "data/aksharantar/test_devanagari.jsonl".to_string();
    let mut model_path = "data/translit_model.bin".to_string();
    let mut vocab_path = "data/word_freq_text.bin".to_string();
    let mut valid_dump: Option<String> = if Path::new("/tmp/dump_valid.jsonl").exists() {
        Some("/tmp/dump_valid.jsonl".to_string())
    } else {
        None
    };
    let mut test_dump: Option<String> = if Path::new("/tmp/dump_test.jsonl").exists() {
        Some("/tmp/dump_test.jsonl".to_string())
    } else {
        None
    };
    let mut out_path = "src/core/reranker_weights.rs".to_string();
    let mut smoke_limit: Option<usize> = None;
    let mut resume = false;
    let mut epochs = 6usize;
    let mut batch_size = 256usize;
    let mut chunk_size = 100_000usize;
    let mut lr = 0.05f64;
    let mut l2 = 1e-4f64;
    let mut lr_sparse: Option<f32> = None;
    let mut l2_sparse: Option<f32> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--train" => train_path = args.next().expect("train path"),
            "--valid" => valid_path = args.next().expect("valid path"),
            "--test" => test_path = args.next().expect("test path"),
            "--valid-dump" => valid_dump = Some(args.next().expect("valid-dump path")),
            "--test-dump" => test_dump = Some(args.next().expect("test-dump path")),
            "--model" => model_path = args.next().expect("model path"),
            "--vocab" => vocab_path = args.next().expect("vocab path"),
            "--out" => out_path = args.next().expect("out path"),
            "--smoke" => smoke_limit = Some(args.next().expect("smoke limit").parse().unwrap()),
            "--chunk-size" => chunk_size = args.next().expect("chunk-size").parse().unwrap(),
            "--epochs" => epochs = args.next().expect("epochs").parse().unwrap(),
            "--batch-size" => batch_size = args.next().expect("batch-size").parse().unwrap(),
            "--lr" => lr = args.next().expect("lr").parse().unwrap(),
            "--l2" => l2 = args.next().expect("l2").parse().unwrap(),
            "--lr-sparse" => lr_sparse = Some(args.next().expect("lr-sparse").parse().unwrap()),
            "--l2-sparse" => l2_sparse = Some(args.next().expect("l2-sparse").parse().unwrap()),
            "--resume" => resume = true,
            other => panic!("unknown argument: {other}"),
        }
    }

    let eff_lr_sparse = lr_sparse.unwrap_or((lr * 0.2) as f32);
    let eff_l2_sparse = l2_sparse.unwrap_or((l2 * 10.0) as f32);

    eprintln!("============================================================");
    eprintln!("        AksharIME Discriminative Reranker Training (W1)     ");
    eprintln!("============================================================");

    // 1. Load model and vocabulary
    let model = TranslitModel::load(Path::new(&model_path)).expect("translit model");
    let decoder = Arc::new(ModelDecoder::with_config(
        model,
        DecoderConfig {
            beam_width: 128,
            ..DecoderConfig::default()
        },
    ));

    let vocab_bytes = std::fs::read(&vocab_path).expect("vocab file");
    let freq: Arc<HashMap<String, u32>> = Arc::new(bincode::deserialize(&vocab_bytes).expect("deserialize vocab"));
    let ranks = Arc::new(FreqRanks::from_freq_map(&freq));
    eprintln!("Loaded model and vocabulary: {} unique words", freq.len());

    let wtrie = Arc::new(akshar_ime::core::wordtrie::WordTrie::from_freq_map(
        &freq,
        &|a| decoder.model.akshara_id(a),
        1,
    ));
    eprintln!("Built WordTrie for candidate union: {} words", wtrie.words);

    // 2. Ingest raw pairs
    let load_pairs = |path: &str, limit: Option<usize>| -> Vec<(String, String)> {
        let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let mut pairs = Vec::new();
        for line in BufReader::new(file).lines().map_while(Result::ok) {
            if let Ok(rec) = serde_json::from_str::<Record>(&line) {
                if !rec.english.trim().is_empty() && !rec.native.trim().is_empty() {
                    pairs.push((rec.english.to_ascii_lowercase(), rec.native));
                    if let Some(lim) = limit {
                        if pairs.len() >= lim {
                            break;
                        }
                    }
                }
            }
        }
        pairs
    };

    let valid_pairs = if valid_dump.is_none() {
        load_pairs(&valid_path, None)
    } else {
        Vec::new()
    };
    let test_pairs = if test_dump.is_none() {
        load_pairs(&test_path, None)
    } else {
        Vec::new()
    };

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // Helper: decode pairs in parallel with Candidate Union (W3)
    let decode_all = |pairs: &[(String, String)], desc: &str| -> Vec<PairCase> {
        let progress = Arc::new(ProgressTracker::new(pairs.len()));
        let chunk_size = pairs.len().div_ceil(threads);
        let results: Vec<PairCase> = std::thread::scope(|s| {
            let mut handles = Vec::new();
            for chunk in pairs.chunks(chunk_size.max(1)) {
                let dec = Arc::clone(&decoder);
                let fr = Arc::clone(&freq);
                let rk = Arc::clone(&ranks);
                let tr = Arc::clone(&wtrie);
                let pg = Arc::clone(&progress);
                handles.push(s.spawn(move || {
                    let mut out = Vec::with_capacity(chunk.len());
                    for (roman, gold) in chunk {
                        if let Some(c) = featurize_candidates(&dec, roman, gold, "AK", &fr, &rk, Some(&tr)) {
                            out.push(c);
                        }
                        pg.inc(1, desc);
                    }
                    out
                }));
            }
            handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        results
    };

    eprintln!("\n--- Step 1: Loading candidate pools ---");
    let valid_cases = if let Some(path) = &valid_dump {
        eprintln!("Loading valid cases from dump: {path}");
        load_dump_cases(path, &freq, &ranks, None).expect("load valid dump")
    } else {
        decode_all(&valid_pairs, "decoding valid")
    };
    eprintln!("Valid cases loaded: {}", valid_cases.len());

    let test_cases = if let Some(path) = &test_dump {
        eprintln!("Loading test cases from dump: {path}");
        load_dump_cases(path, &freq, &ranks, None).expect("load test dump")
    } else {
        decode_all(&test_pairs, "decoding test")
    };
    eprintln!("Test cases loaded: {}", test_cases.len());

    // Baseline heuristic metrics
    let score_heuristic = |cases: &[PairCase]| -> (f64, usize, usize) {
        let mut hit = 0;
        let mut tot = 0;
        for c in cases {
            if c.cands.is_empty() {
                continue;
            }
            tot += 1;
            // Candidate with minimum heur score (lower = better)
            let mut best_i = 0;
            let mut best_h = f64::INFINITY;
            for (i, f) in c.feats.iter().enumerate() {
                if f.dense[4] < best_h {
                    best_h = f.dense[4];
                    best_i = i;
                }
            }
            if c.gold_idx == Some(best_i) {
                hit += 1;
            }
        }
        (hit as f64 / tot.max(1) as f64 * 100.0, hit, tot)
    };

    let (val_base, _, _) = score_heuristic(&valid_cases);
    let (test_base, _, _) = score_heuristic(&test_cases);
    eprintln!("\nBaseline Heuristic: Valid top-1 = {:.2}%, Test top-1 = {:.2}%", val_base, test_base);

    // 3. Normalization parameters computed from valid cases
    eprintln!("\n--- Step 2: Computing standardization statistics ---");
    let mut sum_dense = [0.0f64; DENSE_DIM];
    let mut sq_dense = [0.0f64; DENSE_DIM];
    let mut count_dense = 0.0f64;

    for c in &valid_cases {
        for f in &c.feats {
            for k in 0..DENSE_DIM {
                sum_dense[k] += f.dense[k];
                sq_dense[k] += f.dense[k] * f.dense[k];
            }
            count_dense += 1.0;
        }
    }
    let mut mean_dense = [0.0f64; DENSE_DIM];
    let mut std_dense = [1.0f64; DENSE_DIM];
    for k in 0..DENSE_DIM {
        mean_dense[k] = sum_dense[k] / count_dense.max(1.0);
        let var = (sq_dense[k] / count_dense.max(1.0)) - (mean_dense[k] * mean_dense[k]);
        std_dense[k] = var.max(1e-8).sqrt();
    }

    // 4. Model parameters
    #[derive(Serialize, Deserialize)]
    struct TrainerState {
        chunk_idx: usize,
        total_trained: usize,
        best_valid_acc: f64,
        step: usize,
        w_dense: [f64; DENSE_DIM],
        m_dense: [f64; DENSE_DIM],
        v_dense: [f64; DENSE_DIM],
        w_sparse: Vec<f32>,
        m_sparse: Vec<f32>,
        v_sparse: Vec<f32>,
    }

    let mut w_dense = [0.0f64; DENSE_DIM];
    let mut m_dense = [0.0f64; DENSE_DIM];
    let mut v_dense = [0.0f64; DENSE_DIM];

    let mut w_sparse: Vec<f32> = vec![0.0f32; HASH_SIZE];
    let mut m_sparse: Vec<f32> = vec![0.0f32; HASH_SIZE];
    let mut v_sparse: Vec<f32> = vec![0.0f32; HASH_SIZE];

    let state_path = "data/reranker_state.bin".to_string();

    let export_weights = |out_p: &str, w_d: &[f64; DENSE_DIM], w_s: &[f32]| {
        let max_sparse = w_s.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let sparse_scale = if max_sparse > 0.0 { 127.0 / max_sparse } else { 1.0 };
        let quantized_sparse: Vec<i8> = w_s.iter().map(|&w| (w * sparse_scale).round().clamp(-128.0, 127.0) as i8).collect();
        let non_zero_sparse = quantized_sparse.iter().filter(|&&w| w != 0).count();

        let sparse_bin_path = "data/reranker_weights_sparse.bin".to_string();
        let u8_bytes: Vec<u8> = quantized_sparse.iter().map(|&b| b as u8).collect();
        let _ = std::fs::write(&sparse_bin_path, &u8_bytes);

        let mut out_file = File::create(out_p).expect("create out file");
        writeln!(out_file, "// File: {}", out_p).unwrap();
        writeln!(out_file, "// GENERATED by src/bin/train/train_reranker.rs — do not edit manually.\n").unwrap();
        writeln!(out_file, "pub const GAMMA: f64 = 0.3;").unwrap();
        writeln!(out_file, "pub const LM_W: f64 = 0.85;").unwrap();
        writeln!(out_file, "pub const VOCAB_W: f64 = 0.75;").unwrap();
        writeln!(out_file, "pub const DENSE_DIM: usize = {};", DENSE_DIM).unwrap();
        writeln!(out_file, "pub const HASH_SIZE: usize = {};", HASH_SIZE).unwrap();
        writeln!(out_file, "pub const SPARSE_SCALE: f64 = {:.8};", 1.0 / (sparse_scale as f64)).unwrap();
        writeln!(out_file, "pub const SPARSE_BIN_PATH: &str = {:?};", sparse_bin_path).unwrap();
        writeln!(out_file, "pub const SPARSE_TABLE: &[u8] = include_bytes!(\"../../data/reranker_weights_sparse.bin\");").unwrap();
        writeln!(out_file, "\npub const W_DENSE: [f64; DENSE_DIM] = {:?};", w_d).unwrap();
        writeln!(out_file, "\npub const MEAN_DENSE: [f64; DENSE_DIM] = {:?};", mean_dense).unwrap();
        writeln!(out_file, "\npub const STD_DENSE: [f64; DENSE_DIM] = {:?};", std_dense).unwrap();
        eprintln!(">>> Checkpoint saved ({} non-zero sparse) to {} and {}", non_zero_sparse, out_p, sparse_bin_path);
    };

    fn zscore(xs: &mut [f64]) {
        let n = xs.len() as f64;
        if n < 2.0 {
            return;
        }
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
        let std = var.sqrt() + 1e-9;
        for x in xs.iter_mut() {
            *x = (*x - mean) / std;
        }
    }

    // Evaluate function with optional z-score blending
    let evaluate_model = |cases: &[PairCase], w_d: &[f64; DENSE_DIM], w_s: Option<&[f32]>, gamma: f64| -> f64 {
        let mut hit = 0;
        let mut tot = 0;
        for c in cases {
            if c.cands.is_empty() {
                continue;
            }
            tot += 1;

            if gamma >= 1.0 {
                let mut best_i = 0;
                let mut best_score = f64::NEG_INFINITY;
                for (i, f) in c.feats.iter().enumerate() {
                    let mut s = 0.0f64;
                    for k in 0..DENSE_DIM {
                        s += w_d[k] * ((f.dense[k] - mean_dense[k]) / std_dense[k]);
                    }
                    if let Some(ws) = w_s {
                        for &idx in &f.sparse {
                            s += ws[idx] as f64;
                        }
                    }
                    if s > best_score {
                        best_score = s;
                        best_i = i;
                    }
                }
                if c.gold_idx == Some(best_i) {
                    hit += 1;
                }
            } else if gamma <= 0.0 {
                let mut best_i = 0;
                let mut best_h = f64::INFINITY;
                for (i, f) in c.feats.iter().enumerate() {
                    if f.dense[4] < best_h {
                        best_h = f.dense[4];
                        best_i = i;
                    }
                }
                if c.gold_idx == Some(best_i) {
                    hit += 1;
                }
            } else {
                let mut heur_vec: Vec<f64> = c.feats.iter().map(|f| f.dense[4]).collect();
                let mut model_vec: Vec<f64> = c.feats.iter().map(|f| {
                    let mut s = 0.0f64;
                    for k in 0..DENSE_DIM {
                        s += w_d[k] * ((f.dense[k] - mean_dense[k]) / std_dense[k]);
                    }
                    if let Some(ws) = w_s {
                        for &idx in &f.sparse {
                            s += ws[idx] as f64;
                        }
                    }
                    s
                }).collect();

                zscore(&mut heur_vec);
                zscore(&mut model_vec);

                let mut best_i = 0;
                let mut best_score = f64::NEG_INFINITY;
                for i in 0..c.feats.len() {
                    let s = (1.0 - gamma) * (-heur_vec[i]) + gamma * model_vec[i];
                    if s > best_score {
                        best_score = s;
                        best_i = i;
                    }
                }
                if c.gold_idx == Some(best_i) {
                    hit += 1;
                }
            }
        }
        hit as f64 / tot.max(1) as f64 * 100.0
    };

    // 5. Training loop
    eprintln!("\n--- Step 3: Streaming Training ---");
    let mut best_valid_acc = 0.0f64;
    let mut best_w_dense = w_dense;
    let mut best_w_sparse = w_sparse.clone();
    let mut step = 0usize;
    let mut total_trained = 0usize;
    let mut chunk_idx = 0usize;

    if resume && Path::new(&state_path).exists() {
        if let Ok(bytes) = std::fs::read(&state_path) {
            if let Ok(st) = bincode::deserialize::<TrainerState>(&bytes) {
                chunk_idx = st.chunk_idx;
                total_trained = st.total_trained;
                best_valid_acc = st.best_valid_acc;
                step = st.step;
                w_dense = st.w_dense;
                m_dense = st.m_dense;
                v_dense = st.v_dense;
                w_sparse = st.w_sparse;
                m_sparse = st.m_sparse;
                v_sparse = st.v_sparse;
                best_w_dense = w_dense;
                best_w_sparse = w_sparse.clone();
                eprintln!(
                    ">>> Successfully resumed from chunk {} ({} pairs processed, best valid: {:.2}%)",
                    chunk_idx, total_trained, best_valid_acc
                );
            }
        }
    }

    let total_target = smoke_limit.unwrap_or(3_588_793);
    let eff_chunk = chunk_size.min(total_target);

    let file = File::open(&train_path).unwrap_or_else(|e| panic!("open {train_path}: {e}"));
    let mut lines_iter = BufReader::new(file).lines().map_while(Result::ok);

    if total_trained > 0 {
        eprintln!("Fast-forwarding stream past {} previously trained pairs...", total_trained);
        let mut skipped = 0usize;
        while skipped < total_trained {
            if lines_iter.next().is_none() {
                break;
            }
            skipped += 1;
        }
        eprintln!("Stream positioned at pair {}.", total_trained + 1);
    }

    while total_trained < total_target {
        let to_fetch = (total_target - total_trained).min(eff_chunk);
        let mut chunk_pairs = Vec::with_capacity(to_fetch);
        for line in lines_iter.by_ref().take(to_fetch) {
            if let Ok(rec) = serde_json::from_str::<Record>(&line) {
                if !rec.english.trim().is_empty() && !rec.native.trim().is_empty() {
                    chunk_pairs.push((rec.english.to_ascii_lowercase(), rec.native));
                }
            }
        }
        if chunk_pairs.is_empty() {
            break;
        }

        chunk_idx += 1;
        let c_len = chunk_pairs.len();
        eprintln!(
            "\n>>> Chunk {} [pairs {}..{} / {} ({:.1}%)]",
            chunk_idx,
            total_trained + 1,
            total_trained + c_len,
            total_target,
            (total_trained + c_len) as f64 / total_target as f64 * 100.0
        );

        let chunk_cases = decode_all(&chunk_pairs, &format!("decoding chunk {chunk_idx}"));
        let trainable: Vec<&PairCase> = chunk_cases.iter().filter(|c| c.gold_idx.is_some()).collect();
        eprintln!(
            "Trainable in chunk: {} / {} ({:.1}%)",
            trainable.len(),
            chunk_cases.len(),
            trainable.len() as f64 / chunk_cases.len().max(1) as f64 * 100.0
        );

        let n_ep = if smoke_limit.is_some() { epochs } else { 1 };
        for ep in 1..=n_ep {
            let mut order: Vec<usize> = (0..trainable.len()).collect();
            for i in (1..order.len()).rev() {
                let j = (i.wrapping_mul(6364136223846793005).wrapping_add(ep)) % (i + 1);
                order.swap(i, j);
            }

            let progress = ProgressTracker::new(trainable.len());
            let mut ep_loss = 0.0f64;
            let mut batches = 0usize;

            for batch_indices in order.chunks(batch_size) {
                let mut g_dense = [0.0f64; DENSE_DIM];
                let mut g_sparse: HashMap<usize, f32> = HashMap::new();
                let mut batch_loss = 0.0f64;

                for &idx in batch_indices {
                    let c = trainable[idx];
                    let y = c.gold_idx.unwrap();
                    let n_cand = c.feats.len();

                    let mut scores = Vec::with_capacity(n_cand);
                    for f in &c.feats {
                        let mut s = 0.0f64;
                        for k in 0..DENSE_DIM {
                            s += w_dense[k] * ((f.dense[k] - mean_dense[k]) / std_dense[k]);
                        }
                        for &h in &f.sparse {
                            s += w_sparse[h] as f64;
                        }
                        scores.push(s);
                    }

                    let max_s = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let mut sum_exp = 0.0f64;
                    let mut probs = Vec::with_capacity(n_cand);
                    for &s in &scores {
                        let p = (s - max_s).exp();
                        probs.push(p);
                        sum_exp += p;
                    }
                    for p in &mut probs {
                        *p /= sum_exp.max(1e-12);
                    }

                    batch_loss -= (probs[y].max(1e-15)).ln();

                    for (i, f) in c.feats.iter().enumerate() {
                        let p = probs[i];
                        let delta = if i == y { p - 1.0 } else { p };
                        for k in 0..DENSE_DIM {
                            g_dense[k] += delta * ((f.dense[k] - mean_dense[k]) / std_dense[k]);
                        }
                        for &h in &f.sparse {
                            *g_sparse.entry(h).or_insert(0.0) += delta as f32;
                        }
                    }
                }

                let b_len = batch_indices.len() as f64;
                ep_loss += batch_loss / b_len;
                batches += 1;
                step += 1;

                let beta1_t = 0.9f64.powi(step.min(1000) as i32);
                let beta2_t = 0.999f64.powi(step.min(1000) as i32);

                for k in 0..DENSE_DIM {
                    let g = (g_dense[k] / b_len) + (l2 * w_dense[k]);
                    m_dense[k] = 0.9 * m_dense[k] + 0.1 * g;
                    v_dense[k] = 0.999 * v_dense[k] + 0.001 * g * g;
                    let m_hat = m_dense[k] / (1.0 - beta1_t);
                    let v_hat = v_dense[k] / (1.0 - beta2_t);
                    w_dense[k] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
                }

                for (h, g_val) in g_sparse {
                    let g = (g_val / b_len as f32) + (eff_l2_sparse * w_sparse[h]);
                    m_sparse[h] = 0.9 * m_sparse[h] + 0.1 * g;
                    v_sparse[h] = 0.999 * v_sparse[h] + 0.001 * g * g;
                    let m_hat = m_sparse[h] / (1.0 - beta1_t as f32);
                    let v_hat = v_sparse[h] / (1.0 - beta2_t as f32);
                    w_sparse[h] -= eff_lr_sparse * m_hat / (v_hat.sqrt() + 1e-8);
                }

                let status = format!("loss: {:.3}", ep_loss / batches as f64);
                progress.inc(batch_indices.len(), &status);
            }
        }

        total_trained += c_len;

        // Evaluate after chunk
        let val_pure = evaluate_model(&valid_cases, &w_dense, Some(&w_sparse), 1.0);
        let val_blend = evaluate_model(&valid_cases, &w_dense, Some(&w_sparse), 0.3);
        let test_pure = evaluate_model(&test_cases, &w_dense, Some(&w_sparse), 1.0);
        let test_blend = evaluate_model(&test_cases, &w_dense, Some(&w_sparse), 0.3);

        eprintln!(
            "Chunk {} done | Valid: pure {:.2}%, blend(0.3) {:.2}% (base {:.2}%) | Test: pure {:.2}%, blend(0.3) {:.2}% (base {:.2}%)",
            chunk_idx, val_pure, val_blend, val_base, test_pure, test_blend, test_base
        );

        let eval_criterion = val_blend.max(val_pure);
        if eval_criterion > best_valid_acc {
            best_valid_acc = eval_criterion;
            best_w_dense = w_dense;
            best_w_sparse = w_sparse.clone();
            export_weights(&out_path, &best_w_dense, &best_w_sparse);
        }

        let state = TrainerState {
            chunk_idx,
            total_trained,
            best_valid_acc,
            step,
            w_dense: best_w_dense,
            m_dense,
            v_dense,
            w_sparse: best_w_sparse.clone(),
            m_sparse: m_sparse.clone(),
            v_sparse: v_sparse.clone(),
        };
        if let Ok(bytes) = bincode::serialize(&state) {
            let _ = std::fs::write(&state_path, bytes);
        }
    }

    eprintln!("\n============================================================");
    eprintln!("Training Complete! Processed {} pairs across {} chunks.", total_trained, chunk_idx);
    eprintln!("Final Best Validation Score: {:.2}%", best_valid_acc);
    export_weights(&out_path, &best_w_dense, &best_w_sparse);
    eprintln!("DONE!");
}
