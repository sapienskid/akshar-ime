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
use akshar_ime::core::reranker::{
    extract_dense_features, extract_sparse_features, rank_candidates, FreqRanks, DENSE_DIM,
    HASH_SIZE,
};
use akshar_ime::core::reranker_weights::{MEAN_DENSE, STD_DENSE, W_DENSE};
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

/// Decode depth and beam used when pre-decoding reranker training samples.
/// These mirror the inference path in `core::engine` — the engine requests
/// `max(count * 4, 50)` candidates with `DECODER_BEAM = 64`.  Training on a
/// narrower lattice than inference produces weights fitted to a candidate
/// distribution that never occurs in production.
const RERANK_DECODE_DEPTH: usize = 50;
const RERANK_DECODE_BEAM: usize = 64;

fn main() {
    let mut pairs_path: Option<PathBuf> = None;
    let mut text_path: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut limit: Option<usize> = None;
    let mut iterations: usize = 10;
    let mut epochs: usize = 3;
    let mut min_freq: u32 = 3;
    let mut reranker_pairs: usize = 100_000;
    let mut wasm_mode = false;
    let mut smoke = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pairs" | "-p" => pairs_path = Some(PathBuf::from(args.next().expect("value for --pairs"))),
            "--text" | "-t" => text_path = Some(PathBuf::from(args.next().expect("value for --text"))),
            "--out" | "-o" => out_path = Some(PathBuf::from(args.next().expect("value for --out"))),
            "--limit" | "-n" => limit = Some(args.next().expect("value for --limit").parse().unwrap()),
            "--iterations" | "-i" => iterations = args.next().expect("value for --iterations").parse().unwrap(),
            "--epochs" | "-e" => epochs = args.next().expect("value for --epochs").parse().unwrap(),
            "--min-freq" => min_freq = args.next().expect("value for --min-freq").parse().unwrap(),
            "--reranker-pairs" => reranker_pairs = args.next().expect("value for --reranker-pairs").parse().unwrap(),
            "--wasm" => wasm_mode = true,
            "--smoke" => smoke = true,
            "-h" | "--help" => {
                println!("Akshar Unified Model Trainer");
                println!("Usage: cargo run --release --bin train -- [options]");
                println!("  --pairs <path>          Parallel word pairs JSONL");
                println!("  --text <path>           Clean running text for vocabulary");
                println!("  --out <path>            Output path (default: data/akshar.model)");
                println!("  --min-freq <n>          Prune vocabulary with freq < n (default: 3)");
                println!("  --reranker-pairs <n>    Number of pairs for reranker training (0 = all pairs, default: 100,000)");
                println!("  --limit <n>             Limit pairs (for rapid prototyping)");
                println!("  --iterations <n>        EM iterations (default: 10)");
                println!("  --epochs <n>            Reranker epochs (default: 3)");
                println!("  --wasm                  Export lightweight WASM web profile");
                println!("  --smoke                 Run fast smoke training validation");
                return;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }

    if wasm_mode && min_freq == 3 {
        min_freq = 5; // Default to more aggressive vocabulary pruning for WASM
    }

    let out_path = out_path.unwrap_or_else(|| {
        if wasm_mode {
            PathBuf::from("data/akshar_wasm.model")
        } else {
            PathBuf::from("data/akshar.model")
        }
    });

    if smoke {
        println!(">>> Running in SMOKE mode (fast validation: 1000 pairs, 3 EM iterations, 1 epoch)");
        limit = Some(1000);
        iterations = 3;
        epochs = 1;
        reranker_pairs = 500;
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

    let raw_vocab_count = vocab_freq.len();
    vocab_freq.retain(|_, &mut c| c >= min_freq);
    let ranks = FreqRanks::from_freq_map(&vocab_freq);
    println!(
        "Compiled clean vocabulary of {} unique words (pruned {} hapax/noise words < {} freq) in {:.2?}.",
        vocab_freq.len(),
        raw_vocab_count.saturating_sub(vocab_freq.len()),
        min_freq,
        vocab_t0.elapsed()
    );

    // -------------------------------------------------------------------------
    // Phase 3: Train Discriminative Reranker Weights
    // -------------------------------------------------------------------------
    println!("\n[Phase 3/4] Training Discriminative Reranker...");
    let rank_t0 = Instant::now();
    let decoder = ModelDecoder::with_config(
        translit_model.clone(),
        DecoderConfig {
            // Match the engine's DECODER_BEAM so the reranker is trained on
            // the candidate sets it will actually have to rank.
            beam_width: RERANK_DECODE_BEAM,
            ..DecoderConfig::default()
        },
    );

    let word_trie = WordTrie::from_freq_map(&vocab_freq, &|a| translit_model.akshara_id(a), 1);

    let num_train_pairs = if smoke {
        500
    } else if reranker_pairs == 0 {
        raw_pairs.len()
    } else {
        raw_pairs.len().min(reranker_pairs)
    };
    println!("Pre-decoding candidates for {} training pairs...", num_train_pairs);

    struct RerankItem {
        target_idx: usize,
        base_scores: Vec<f64>,
        sparse: Vec<Vec<usize>>,
    }

    let decode_t0 = Instant::now();
    let mut samples: Vec<RerankItem> = Vec::with_capacity(num_train_pairs);

    for (roman, gold) in &raw_pairs[..num_train_pairs] {
        // Decode at the same depth the engine uses at inference time
        // (engine.rs requests max(count*4, 50) with DECODER_BEAM = 64).  A
        // shallower training decode changes which candidates compete, and
        // therefore what the softmax is trained to discriminate.
        let cands = decoder.decode_union(roman, RERANK_DECODE_DEPTH, Some(&word_trie));
        // Candidate order, heuristic and heuristic rank must be computed the
        // same way as at inference — see reranker::rank_candidates.
        let (order, heur, heur_rank) = rank_candidates(&cands, &vocab_freq);
        if let Some(target_idx) = order.iter().position(|c| c.dev == *gold) {
            let n_cand = order.len();
            let cand_sparse: Vec<Vec<usize>> = order.iter().map(|c| {
                let aks = akshar_ime::core::akshara::segment(&c.dev);
                extract_sparse_features(&c.dev, roman, c.akshara_count, &aks)
            }).collect();

            let mut base_scores = Vec::with_capacity(n_cand);
            for (idx, c) in order.iter().enumerate() {
                let dense = extract_dense_features(c, idx, heur[idx], heur_rank[idx], roman, &vocab_freq, &ranks);
                let mut score = 0.0f64;
                for k in 0..DENSE_DIM {
                    score += W_DENSE[k] * ((dense[k] - MEAN_DENSE[k]) / STD_DENSE[k]);
                }
                base_scores.push(score);
            }

            samples.push(RerankItem {
                target_idx,
                base_scores,
                sparse: cand_sparse,
            });
        }
    }
    println!(
        "Pre-decoded {} valid samples with targets in candidate list in {:.2?}.",
        samples.len(),
        decode_t0.elapsed()
    );

    let mut sparse_table: Vec<f32> = vec![0.0f32; HASH_SIZE];
    let mut grad_sq: Vec<f32> = vec![0.0f32; HASH_SIZE];
    let mut lr = 0.05f64;

    for ep in 1..=epochs {
        let mut ep_loss = 0.0f64;
        let mut ep_hits = 0usize;

        for s in &samples {
            let mut scores = s.base_scores.clone();

            for (idx, sparse_feats) in s.sparse.iter().enumerate() {
                for &h in sparse_feats {
                    scores[idx] += sparse_table[h] as f64;
                }
            }

            let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_s: Vec<f64> = scores.iter().map(|&sc| (sc - max_s).exp()).collect();
            let sum_exp: f64 = exp_s.iter().sum();
            let probs: Vec<f64> = exp_s.iter().map(|&e| e / (sum_exp + 1e-12)).collect();

            ep_loss += -probs[s.target_idx].max(1e-12).ln();
            if probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 == s.target_idx {
                ep_hits += 1;
            }

            for (idx, p) in probs.iter().enumerate() {
                let grad = if idx == s.target_idx { *p - 1.0 } else { *p };
                if grad.abs() > 1e-5 {
                    for &h in &s.sparse[idx] {
                        let g = grad as f32;
                        grad_sq[h] += g * g;
                        let eff_lr = (lr as f32) / (grad_sq[h].sqrt() + 1e-4);
                        sparse_table[h] -= eff_lr * g;
                    }
                }
            }
        }
        lr *= 0.8;
        if !samples.is_empty() {
            println!(
                "  Epoch {}/{}: loss={:.4}, top-1 accuracy={:.2}%",
                ep,
                epochs,
                ep_loss / samples.len() as f64,
                ep_hits as f64 / samples.len() as f64 * 100.0
            );
        }
    }
    println!("Reranker trained in {:.2?}.", rank_t0.elapsed());

    // -------------------------------------------------------------------------
    // Phase 4: Pack Unified Model
    // -------------------------------------------------------------------------
    println!("\n[Phase 4/4] Packing Unified Model -> {} ...", out_path.display());
    let pack_t0 = Instant::now();

    let max_sparse = sparse_table.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
    let sparse_scale = if max_sparse > 0.0 { 127.0 / max_sparse } else { 1.0 };
    let quantized_table: Vec<i8> = sparse_table
        .iter()
        .map(|&w| (w * sparse_scale).round().clamp(-128.0, 127.0) as i8)
        .collect();
    let non_zero = quantized_table.iter().filter(|&&w| w != 0).count();
    println!("Quantized sparse table: {} non-zero weights (scale: {:.4})", non_zero, sparse_scale);

    println!("Pruning unreferenced aksharas and cleaning non-Nepali transitions...");
    let mut seen_aks = std::collections::HashSet::new();
    for word in vocab_freq.keys() {
        for a in akshar_ime::core::akshara::segment(word) {
            if let Some(id) = translit_model.akshara_id(&a) {
                seen_aks.insert(id);
            }
        }
    }
    let before_em = translit_model.emissions.iter().filter(|e| !e.is_empty()).count();
    for (a, em) in translit_model.emissions.iter_mut().enumerate() {
        if !seen_aks.contains(&(a as u32)) {
            em.clear();
        }
    }
    for (a, bi) in translit_model.bigrams.iter_mut().enumerate() {
        if !seen_aks.contains(&(a as u32)) {
            bi.clear();
        } else {
            bi.retain(|(next_id, _)| seen_aks.contains(next_id));
        }
    }
    let mut new_keys = Vec::new();
    let mut new_trigrams = Vec::new();
    let mut new_backoff = Vec::new();
    for (i, &(a, b)) in translit_model.trigram_keys.iter().enumerate() {
        if seen_aks.contains(&a) && seen_aks.contains(&b) {
            let mut list = translit_model.trigrams[i].clone();
            list.retain(|(c, _)| seen_aks.contains(c));
            if !list.is_empty() {
                new_keys.push((a, b));
                new_trigrams.push(list);
                new_backoff.push(translit_model.trigram_backoff.get(i).copied().unwrap_or(0.0));
            }
        }
    }
    translit_model.trigram_keys = new_keys;
    translit_model.trigrams = new_trigrams;
    translit_model.trigram_backoff = new_backoff;
    translit_model.build_trigram_index();
    let after_em = translit_model.emissions.iter().filter(|e| !e.is_empty()).count();
    println!("Pruned unused emission rows: {} -> {} (and cleaned transitions)", before_em, after_em);

    // Carry the run's own quantization scale: the table is meaningless
    // without it (see UnifiedModel::sparse_scale).
    let unified = UnifiedModel::new(
        translit_model,
        quantized_table,
        1.0 / sparse_scale as f64,
        vocab_freq,
    );
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
