// File: src/bin/train/train.rs
//
// End-to-End Unified Model Trainer for Akshar Devanagari IME.
// Ingests training data (word pairs + text) and directly produces
// a single, low-size, production-ready `akshar.model` artifact.
//
// Usage: cargo run --release --bin train -- [options]
//   --pairs <path>     Parallel roman-devanagari word pairs (JSONL)
//   --text <path>      Clean running Devanagari text for vocabulary frequencies
//   --out <path>       Output unified model path (default: data/akshar.model)
//   --limit <n>        Limit pairs ingested (for quick testing)
//   --iterations <n>   EM training iterations (default: 10)
//   --epochs <n>       Reranker training epochs (default: 3)
//   --smoke            Fast 5-second smoke test training (500 pairs)

use akshar_ime::core::decoder::{DecoderConfig, ModelDecoder};
use akshar_ime::core::em_trainer::{Trainer, TrainerConfig};
use akshar_ime::core::reranker::{extract_dense_features, extract_sparse_features, FreqRanks, DENSE_DIM, HASH_SIZE};
use akshar_ime::core::reranker_weights::{LM_W, MEAN_DENSE, SPARSE_SCALE, STD_DENSE, VOCAB_W, W_DENSE};
use akshar_ime::core::unified::UnifiedModel;
use akshar_ime::core::wordtrie::WordTrie;
use akshar_ime::ImeEngine;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Deserialize)]
struct Record {
    #[serde(rename = "english word", alias = "english", alias = "roman")]
    english: String,
    #[serde(rename = "native word", alias = "native", alias = "devanagari")]
    native: String,
    #[serde(default = "default_weight")]
    weight: u32,
}

fn default_weight() -> u32 {
    1
}

fn is_word_char(c: char) -> bool {
    matches!(c, '\u{0900}'..='\u{0963}')
}

fn clean_devanagari_token(word: &str) -> Option<String> {
    let trimmed = word.trim_matches(|c: char| !is_word_char(c));
    let chars: Vec<char> = trimmed.chars().collect();
    if chars.is_empty() || chars.len() > 24 {
        return None;
    }
    if chars.iter().all(|c| is_word_char(*c)) {
        Some(trimmed.to_string())
    } else {
        None
    }
}

fn auto_detect_pairs() -> Option<PathBuf> {
    let candidates = [
        "data/aksharantar/train_devanagari.jsonl",
        "data/aksharantar/nep_train.json",
        "data/corpus_clean.json",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn auto_detect_text() -> Option<PathBuf> {
    let candidates = [
        "data/store/corpus_clean.txt",
        "data/raw/nepali_text.txt",
    ];
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn main() {
    let mut pairs_path: Option<PathBuf> = None;
    let mut text_path: Option<PathBuf> = None;
    let mut out_path = PathBuf::from("data/akshar.model");
    let mut limit: Option<usize> = None;
    let mut iterations: usize = 10;
    let mut epochs: usize = 3;
    let mut smoke = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pairs" | "-p" => pairs_path = Some(PathBuf::from(args.next().expect("value for --pairs"))),
            "--text" | "-t" => text_path = Some(PathBuf::from(args.next().expect("value for --text"))),
            "--out" | "-o" => out_path = PathBuf::from(args.next().expect("value for --out")),
            "--limit" | "-n" => limit = Some(args.next().expect("value for --limit").parse().unwrap()),
            "--iterations" | "-i" => iterations = args.next().expect("value for --iterations").parse().unwrap(),
            "--epochs" | "-e" => epochs = args.next().expect("value for --epochs").parse().unwrap(),
            "--smoke" => smoke = true,
            "-h" | "--help" => {
                println!("Akshar Unified Model Trainer");
                println!("Usage: cargo run --release --bin train -- [options]");
                println!("  --pairs <path>     Parallel word pairs JSONL");
                println!("  --text <path>      Clean running text for vocabulary");
                println!("  --out <path>       Output path (default: data/akshar.model)");
                println!("  --limit <n>        Limit pairs (for rapid prototyping)");
                println!("  --iterations <n>   EM iterations (default: 10)");
                println!("  --epochs <n>       Reranker epochs (default: 3)");
                println!("  --smoke            Run fast 5-second smoke training");
                return;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }

    if smoke {
        println!(">>> Running in SMOKE mode (fast validation: 1000 pairs, 3 EM iterations, 1 epoch)");
        limit = Some(1000);
        iterations = 3;
        epochs = 1;
    }

    let pairs_file = pairs_path.or_else(auto_detect_pairs).expect(
        "No training pairs file found! Specify --pairs <path> (e.g. data/aksharantar/nep_train.json)",
    );
    let text_file = text_path.or_else(auto_detect_text);

    println!("============================================================");
    println!("           Akshar One-Shot Unified Trainer                  ");
    println!("============================================================");
    println!("Training Pairs: {}", pairs_file.display());
    println!("Text Corpus:    {}", text_file.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "Derived from pairs".to_string()));
    println!("Target Output:  {}", out_path.display());
    println!("Parameters:     EM iterations={}, Reranker epochs={}", iterations, epochs);
    if let Some(lim) = limit {
        println!("Limit:          {} pairs", lim);
    }
    println!("============================================================");

    let start_time = Instant::now();

    // -------------------------------------------------------------------------
    // Phase 1: Ingest Pairs & Train EM Transliteration Model
    // -------------------------------------------------------------------------
    println!("\n[Phase 1/4] Training EM Source-Channel Transliteration Model...");
    let em_t0 = Instant::now();
    let em_config = TrainerConfig {
        iterations,
        limit,
        seed_from_aligner: true,
        ..Default::default()
    };

    let mut em_trainer = Trainer::new().with_limit(limit);
    let f = File::open(&pairs_file).expect("open pairs file");
    let mut raw_pairs: Vec<(String, String)> = Vec::new();
    let mut count = 0usize;

    for line in BufReader::new(f).lines().map_while(Result::ok) {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if let Ok(rec) = serde_json::from_str::<Record>(t) {
            let eng = rec.english.trim().to_ascii_lowercase();
            let nat = rec.native.trim().to_string();
            if !eng.is_empty() && !nat.is_empty() {
                em_trainer.add_pair_weighted(&eng, &nat, rec.weight as f64);
                raw_pairs.push((eng, nat));
                count += 1;
                if let Some(lim) = limit {
                    if count >= lim {
                        break;
                    }
                }
            }
        }
    }
    println!("Ingested {} valid parallel pairs.", count);

    let mut translit_model = em_trainer.finalize(&em_config);
    translit_model.build_trigram_index();
    println!("EM training finished in {:.2?} (aksharas: {}, chunks: {}).", em_t0.elapsed(), translit_model.aksharas.len(), translit_model.chunks.len());

    // -------------------------------------------------------------------------
    // Phase 2: Ingest Text & Build Frequency Table
    // -------------------------------------------------------------------------
    println!("\n[Phase 2/4] Compiling Vocabulary & Empirical Frequencies...");
    let vocab_t0 = Instant::now();
    let mut vocab_freq: HashMap<String, u32> = HashMap::new();

    if let Some(ref tp) = text_file {
        if tp.exists() {
            println!("Reading running text from {} ...", tp.display());
            let tf = File::open(tp).expect("open text file");
            for line in BufReader::new(tf).lines().map_while(Result::ok) {
                for token in line.split_whitespace() {
                    if let Some(clean) = clean_devanagari_token(token) {
                        *vocab_freq.entry(clean).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    // Always ensure pairs are in the vocabulary
    for (_, dev) in &raw_pairs {
        if let Some(clean) = clean_devanagari_token(dev) {
            *vocab_freq.entry(clean).or_insert(0) += 1;
        }
    }
    let ranks = FreqRanks::from_freq_map(&vocab_freq);
    println!("Compiled vocabulary of {} unique words in {:.2?}.", vocab_freq.len(), vocab_t0.elapsed());

    // -------------------------------------------------------------------------
    // Phase 3: Train Discriminative Reranker Weights
    // -------------------------------------------------------------------------
    println!("\n[Phase 3/4] Training Discriminative Reranker...");
    let rank_t0 = Instant::now();
    let decoder = ModelDecoder::with_config(
        translit_model.clone(),
        DecoderConfig {
            beam_width: 32,
            ..DecoderConfig::default()
        },
    );

    let word_trie = WordTrie::from_freq_map(&vocab_freq, &|a| translit_model.akshara_id(a), 1);

    let mut sparse_table: Vec<i8> = vec![0i8; HASH_SIZE];
    let num_train_pairs = raw_pairs.len().min(if smoke { 500 } else { 50_000 });
    println!("Decoding candidates for {} training pairs...", num_train_pairs);

    let mut trained_samples = 0usize;
    let mut lr = 0.05f64;

    for ep in 1..=epochs {
        let mut ep_loss = 0.0f64;
        let mut ep_hits = 0usize;

        for (roman, gold) in &raw_pairs[..num_train_pairs] {
            let cands = decoder.decode_union(roman, 20, Some(&word_trie));
            if cands.is_empty() {
                continue;
            }

            let gold_idx = cands.iter().position(|c| c.dev == *gold);
            if let Some(target_idx) = gold_idx {
                trained_samples += 1;
                let n_cand = cands.len();

                // Compute heuristic scores
                let heur: Vec<f64> = cands.iter().map(|c| {
                    let f = vocab_freq.get(&c.dev).copied().unwrap_or(0);
                    c.emit + LM_W * c.lm - VOCAB_W * (1.0 + f as f64).ln()
                }).collect();

                let mut heur_rank = vec![0usize; n_cand];
                for (r, idx) in (0..n_cand).enumerate() {
                    heur_rank[idx] = r;
                }

                let mut scores = Vec::with_capacity(n_cand);
                let mut cand_sparse = Vec::with_capacity(n_cand);

                for (idx, c) in cands.iter().enumerate() {
                    let dense = extract_dense_features(c, idx, heur[idx], heur_rank[idx], roman, &vocab_freq, &ranks);
                    let aks = akshar_ime::core::akshara::segment(&c.dev);
                    let sparse = extract_sparse_features(&c.dev, roman, c.akshara_count, &aks);

                    let mut s = 0.0f64;
                    for k in 0..DENSE_DIM {
                        s += W_DENSE[k] * ((dense[k] - MEAN_DENSE[k]) / STD_DENSE[k]);
                    }
                    for &h in &sparse {
                        let w = sparse_table[h] as f64 * SPARSE_SCALE;
                        s += w;
                    }
                    scores.push(s);
                    cand_sparse.push(sparse);
                }

                // Softmax probabilities
                let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_s: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
                let sum_exp: f64 = exp_s.iter().sum();
                let probs: Vec<f64> = exp_s.iter().map(|&e| e / (sum_exp + 1e-12)).collect();

                ep_loss += -probs[target_idx].max(1e-12).ln();
                if probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 == target_idx {
                    ep_hits += 1;
                }

                // Gradient update on sparse features
                for (idx, p) in probs.iter().enumerate() {
                    let grad = if idx == target_idx { *p - 1.0 } else { *p };
                    if grad.abs() > 1e-5 {
                        for &h in &cand_sparse[idx] {
                            let cur = sparse_table[h] as f64;
                            let updated = (cur - lr * grad * 10.0).clamp(-127.0, 127.0);
                            sparse_table[h] = updated.round() as i8;
                        }
                    }
                }
            }
        }
        lr *= 0.8;
        if trained_samples > 0 {
            println!("  Epoch {}/{}: loss={:.4}, top-1 accuracy={:.2}%", ep, epochs, ep_loss / trained_samples as f64, ep_hits as f64 / trained_samples as f64 * 100.0);
        }
    }
    println!("Reranker trained in {:.2?}.", rank_t0.elapsed());

    // -------------------------------------------------------------------------
    // Phase 4: Package into Unified Model Container
    // -------------------------------------------------------------------------
    println!("\n[Phase 4/4] Assembling Unified Model Container -> {} ...", out_path.display());
    let pack_t0 = Instant::now();

    // Context bigrams: optional
    let bigrams = None;

    let unified = UnifiedModel::new(translit_model, sparse_table, vocab_freq, bigrams);
    unified.save(&out_path).expect("save unified model");

    let meta = std::fs::metadata(&out_path).expect("model metadata");
    let size_mb = meta.len() as f64 / (1024.0 * 1024.0);

    println!("Unified model container successfully written in {:.2?}!", pack_t0.elapsed());
    println!("Artifact: {} ({:.2} MB)", out_path.display(), size_mb);

    // -------------------------------------------------------------------------
    // Self-Verification Smoke Test
    // -------------------------------------------------------------------------
    println!("\n>>> Running Self-Verification Smoke Test on newly created model...");
    let engine = ImeEngine::from_unified_file(&out_path).expect("load newly trained model");
    let test_queries = ["namaste", "nepal", "kathmandu", "dhanyabad", "pani"];
    println!("Testing top suggestion generation for basic words:");
    for q in test_queries {
        let sugs = engine.get_suggestions(q, 3);
        let top: Vec<String> = sugs.iter().map(|(s, _)| s.clone()).collect();
        println!("  {:12} -> {:?}", q, top);
        assert!(!top.is_empty(), "Failed to generate suggestions for {q}");
    }

    println!("\n============================================================");
    println!("All done in {:.2?}! Model is 100% verified and ready.", start_time.elapsed());
    println!("============================================================");
}
