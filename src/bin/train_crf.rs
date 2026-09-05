// File: src/bin/train_crf.rs
//
// S3: train the conditional random field over the akshara lattice on the
// cleaned Devanagari pair corpus, save data/crf_model.bin, and run a
// training-set decode smoke gate.
//
// The smoke gate decodes a sample of the TRAINING pairs after training and
// refuses to report success below the threshold — a converged log-likelihood
// with a broken decode is exactly the failure mode this guards against.
//
// Usage:
//   cargo run --release --bin train_crf -- \
//     [--train data/aksharantar/train_devanagari.jsonl] \
//     [--pairs 100000] [--epochs 2] [--lr 0.4] [--l2 1e-4] \
//     [--model data/translit_model.bin] [--out data/crf_model.bin] \
//     [--smoke 300]
use akshar_ime::core::akshara;
use akshar_ime::core::crf::CrfTrainer;
use akshar_ime::core::decoder::{DecoderConfig, ModelDecoder};
use akshar_ime::core::translit_model::TranslitModel;
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::sync::Arc;

#[derive(Deserialize)]
struct Record {
    #[serde(rename = "english word", alias = "english")]
    english: String,
    #[serde(rename = "native word", alias = "native")]
    native: String,
}

/// Deterministic LCG shuffle — no rand dependency, reproducible runs.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = (self.next() as usize) % (i + 1);
            xs.swap(i, j);
        }
    }
}

fn main() {
    let mut train = "data/aksharantar/train_devanagari.jsonl".to_string();
    let mut model_path = "data/translit_model.bin".to_string();
    let mut out = "data/crf_model.bin".to_string();
    let mut pairs = 100_000usize;
    let mut epochs = 2usize;
    let mut lr = 0.1f64;
    let mut l2 = 1e-3f64;
    let mut smoke = 300usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--train" => train = args.next().expect("path"),
            "--model" => model_path = args.next().expect("path"),
            "--out" => out = args.next().expect("path"),
            "--pairs" => pairs = args.next().expect("n").parse().unwrap(),
            "--epochs" => epochs = args.next().expect("n").parse().unwrap(),
            "--lr" => lr = args.next().expect("x").parse().unwrap(),
            "--l2" => l2 = args.next().expect("x").parse().unwrap(),
            "--smoke" => smoke = args.next().expect("n").parse().unwrap(),
            other => panic!("unknown arg {other}"),
        }
    }

    let model = TranslitModel::load(std::path::Path::new(&model_path)).expect("model");
    eprintln!(
        "model: aksharas={} chunks={}",
        model.aksharas.len(),
        model.chunks.len()
    );

    // Sample the corpus.
    let file = std::fs::File::open(&train).expect("train file");
    let mut all: Vec<(String, String)> = BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| {
            serde_json::from_str::<Record>(&l)
                .ok()
                .map(|r| (r.english, r.native))
        })
        .collect();
    let total_available = all.len();
    Lcg(0x9E3779B97F4A7C15).shuffle(&mut all);
    all.truncate(pairs);
    eprintln!("sampled {} of {} pairs", all.len(), total_available);

    // Chunk -> akshara candidates, sorted by base emission weight.
    let mut reverse_weights: HashMap<String, Vec<(u32, f32)>> = HashMap::new();
    for (a, list) in model.emissions.iter().enumerate() {
        for &(cid, w) in list {
            if w < 15.0 {
                if let Some(chunk) = model.chunks.get(cid as usize) {
                    reverse_weights
                        .entry(chunk.clone())
                        .or_default()
                        .push((a as u32, w));
                }
            }
        }
    }
    let mut candidates_of: HashMap<String, Vec<u32>> = HashMap::new();
    for (chunk, mut v) in reverse_weights {
        v.sort_by(|a, b| a.1.total_cmp(&b.1));
        v.truncate(16);
        candidates_of.insert(chunk, v.into_iter().map(|(a, _)| a).collect());
    }
    eprintln!("candidate index: {} chunks", candidates_of.len());

    let chunk_to_model_cid: HashMap<&str, u32> = model
        .chunks
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as u32))
        .collect();

    let mut trainer = CrfTrainer::new(lr, l2);
    for chunk in candidates_of.keys() {
        trainer.chunk_id(chunk);
    }

    // Segment natives to akshara id sequences; the lattice has only
    // positive-length chunks, so pairs whose native has more aksharas than
    // the roman has characters cannot align and are skipped.
    let mut data: Vec<(String, Vec<u32>)> = Vec::with_capacity(all.len());
    for (roman, native) in &all {
        if let Ok(aks) = akshara::segment(native)
            .into_iter()
            .map(|a| model.akshara_id(&a).ok_or(()))
            .collect::<Result<Vec<u32>, ()>>()
        {
            if !aks.is_empty() && aks.len() <= roman.chars().count() {
                let rl = roman.to_ascii_lowercase();
                // pre-register every chunk of every usable pair (pair_gradient
                // is pure and cannot register)
                let m = rl.len();
                for i in 0..m {
                    for l in 1..=akshar_ime::core::crf::MAX_CHUNK.min(m - i) {
                        trainer.chunk_id(&rl[i..i + l]);
                    }
                }
                data.push((rl, aks));
            }
        }
    }
    eprintln!("{} usable pairs after segmentation", data.len());

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    eprintln!("training on {threads} threads (minibatch-parallel SGD)");

    let t0 = std::time::Instant::now();
    let minibatch = 256usize;
    for epoch in 0..epochs {
        let mut order: Vec<usize> = (0..data.len()).collect();
        Lcg(0xDEADBEEF + epoch as u64).shuffle(&mut order);
        let mut sum = 0.0f64;
        let mut skipped = 0usize;

        for mb in order.chunks(minibatch) {
            let slice_len = mb.len().div_ceil(threads);
            let results = std::thread::scope(|scope| {
                let data_ref = &data;
                let trainer_ref = &trainer;
                let cand_ref = &candidates_of;
                let model_ref = &model;
                let chunk_cid_ref = &chunk_to_model_cid;
                let handles: Vec<_> = mb
                    .chunks(slice_len.max(1))
                    .map(|slice: &[usize]| {
                        scope.spawn(move || {
                            let mut ll = 0.0f64;
                            let mut grad =
                                akshar_ime::core::crf::Grad::default();
                            let base_fn = |prev: Option<u32>, a: u32, chunk: &str| -> f64 {
                                let emit_w = if let Some(&cid) = chunk_cid_ref.get(chunk) {
                                    model_ref
                                        .emissions
                                        .get(a as usize)
                                        .and_then(|list| list.iter().find(|(c, _)| *c == cid))
                                        .map(|(_, w)| *w as f64)
                                        .unwrap_or(f64::INFINITY)
                                } else {
                                    f64::INFINITY
                                };
                                let lm_w = match prev {
                                    None => model_ref.start_weight(a),
                                    Some(p) => model_ref.bigram_weight(p, a),
                                };
                                -(emit_w + lm_w)
                            };
                            for &idx in slice {
                                let (roman, aks) = &data_ref[idx];
                                if let Some((l, g)) = trainer_ref.pair_gradient_with_base(
                                    roman,
                                    aks,
                                    &|s: &str| {
                                        cand_ref.get(s).cloned().unwrap_or_default()
                                    },
                                    Some(&base_fn),
                                ) {
                                    ll += l;
                                    grad.merge(g);
                                } else {
                                    ll += f64::NAN;
                                }
                            }
                            (ll, grad)
                        })
                    })
                    .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("worker panicked"))
                    .collect::<Vec<_>>()
            });
            let mut batch_grad = akshar_ime::core::crf::Grad::default();
            for (ll, grad) in results {
                if ll.is_finite() {
                    sum += ll;
                } else {
                    skipped += 1;
                }
                batch_grad.merge(grad);
            }
            trainer.apply_gradient(&batch_grad);
        }

        eprintln!(
            "epoch {} done: avg loglik {:.4}, skipped {} ({:.0}s)",
            epoch + 1,
            sum / data.len() as f64,
            skipped,
            t0.elapsed().as_secs_f64()
        );
        trainer.lr *= 0.6;
    }

    // Sanity: the trained log-likelihood must never be positive (the gold
    // path is inside the total lattice under one shared measure).
    // (train_pair debug_asserts this in debug builds.)

    {
        let max_emit = trainer.model.emit.values().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_emit = trainer.model.emit.values().copied().fold(f64::INFINITY, f64::min);
        let max_trans = trainer.model.trans.values().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_trans = trainer.model.trans.values().copied().fold(f64::INFINITY, f64::min);
        eprintln!("emit weight range: [{:.2}, {:.2}], trans weight range: [{:.2}, {:.2}]",
            min_emit, max_emit, min_trans, max_trans);
    }

    let bytes = bincode::serialize(&trainer.model).expect("serialize");
    std::fs::write(&out, &bytes).expect("write");
    eprintln!(
        "saved {}: emit {} trans {} start {} chunks {} ({:.1} MB)",
        out,
        trainer.model.emit.len(),
        trainer.model.trans.len(),
        trainer.model.start.len(),
        trainer.model.chunk_ids.len(),
        bytes.len() as f64 / 1e6
    );

    // ---- Smoke gate: decode training pairs with the trained CRF. ----
    if smoke > 0 {
        let crf = Arc::new(trainer.model);
        let dec = ModelDecoder::with_config(
            model,
            DecoderConfig {
                beam_width: 64,
                crf: Some(crf),
                ..DecoderConfig::default()
            },
        );
        let mut hit = 0usize;
        let mut total = 0usize;
        for (idx, (roman, aks)) in data.iter().take(smoke).enumerate() {
            // Rebuild the native word from the akshara ids for comparison.
            let native: String = aks.iter().map(|&id| dec.model.aksharas[id as usize].clone()).collect();
            if idx == 0 {
                let detailed = dec.decode_detailed(roman, 5);
                eprintln!("detailed for {}:", roman);
                for c in &detailed {
                    eprintln!("  dev='{}' emit={} lm={} score={}", c.dev, c.emit, c.lm, c.emit + c.lm * dec.config.lm_weight);
                }
            }
            let top = dec.decode(roman, 1);
            if idx < 5 {
                eprintln!("smoke {}: roman='{}' native='{}' top={:?}", idx, roman, native, top.first());
            }
            total += 1;
            hit += top.first().is_some_and(|(d, _)| *d == native) as usize;
        }
        let pct = hit as f64 / total.max(1) as f64 * 100.0;
        eprintln!("SMOKE GATE: training-set decode top-1 = {hit}/{total} = {pct:.1}%");
        if pct < 50.0 {
            eprintln!("SMOKE GATE FAILED — the decode path does not match the trained model; refusing to bless the artifact.");
            std::process::exit(1);
        }
        eprintln!("SMOKE GATE PASSED");
    }
}
