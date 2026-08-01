// File: src/bin/train_reranker.rs
//
// MERT weight training for the discriminative reranker.
//
// Decodes the dev split (data/aksharantar/nep_valid.json) into k-best lists,
// precomputes feature vectors, then runs coordinate ascent over the feature
// weights to directly maximise top-1 accuracy.
//
// Usage:
//   cargo run --release --bin train_reranker -- [options]
//     --model <path>   translit model (default: data/translit_model.bin)
//     --dev <path>     dev split (default: data/aksharantar/nep_valid.json)
//     --out <path>     output weights file (default: data/reranker_weights.json)

use akshar_ime::core::decoder::{DecoderConfig, ModelDecoder};
use akshar_ime::core::lexicon::RomanLexicon;
use akshar_ime::core::reranker::{Reranker, NUM_FEATURES};
use akshar_ime::core::translit_model::TranslitModel;
use serde::Deserialize;
use serde_json::json;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Deserialize)]
struct Record<'a> {
    #[serde(rename = "english word")]
    roman: &'a str,
    #[serde(rename = "native word")]
    target: &'a str,
}

struct DevCase {
    cands: Vec<[f64; NUM_FEATURES]>,
    target_pos: Option<usize>,
}

fn main() {
    let mut model_path = "data/translit_model.bin".to_string();
    let mut dev_path = "data/aksharantar/nep_valid.json".to_string();
    let mut out_path = "data/reranker_weights.json".to_string();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_path = next_value(&arg, args.next()),
            "--dev" => dev_path = next_value(&arg, args.next()),
            "--out" => out_path = next_value(&arg, args.next()),
            "--help" | "-h" => {
                println!("usage: train_reranker [--model p] [--dev p] [--out p]");
                return;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }

    let model = TranslitModel::load(Path::new(&model_path)).expect("load model");
    let decoder = ModelDecoder::with_config(
        model,
        DecoderConfig {
            beam_width: 128,
            ..Default::default()
        },
    );
    let lexicon = RomanLexicon::load(Path::new("data/roman_lexicon.bin")).ok();
    let featurizer = Reranker::default().with_lexicon(lexicon);
    // Precompute candidate feature vectors over the dev split.
    let cases = build_dev_cases(&dev_path, &decoder, &featurizer);
    let total = cases.len();
    let in_kbest = cases.iter().filter(|c| c.target_pos.is_some()).count();
    eprintln!(
        "dev cases: {total}, target in top-50: {in_kbest} ({:.1}%)",
        in_kbest as f64 / total as f64 * 100.0
    );

    let mut weights = [1.0f64; NUM_FEATURES];
    weights[0] = 1.0;
    weights[1] = 1.0;
    let baseline = evaluate(&weights, &cases);
    eprintln!(
        "baseline top-1: {baseline}/{total} ({:.2}%)",
        baseline as f64 / total as f64 * 100.0
    );

    let grids: [Vec<f64>; NUM_FEATURES] = [
        vec![0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0],
        vec![0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0],
        vec![-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0],
        vec![0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0],
    ];

    let mut best_acc = evaluate(&weights, &cases);
    for pass in 0..6 {
        for i in 0..NUM_FEATURES {
            let mut best_w = weights[i];
            let mut best_i_acc = best_acc;
            for &w in &grids[i] {
                if (w - weights[i]).abs() < 1e-9 {
                    continue;
                }
                weights[i] = w;
                let acc = evaluate(&weights, &cases);
                if acc > best_i_acc {
                    best_i_acc = acc;
                    best_w = w;
                }
            }
            weights[i] = best_w;
            best_acc = best_i_acc;
        }
        eprintln!(
            "pass {pass}: top-1 {best_acc}/{total} ({:.2}%)  weights={:?}",
            best_acc as f64 / total as f64 * 100.0,
            weights
        );
    }

    let names = akshar_ime::core::reranker::feature_names();
    let obj = serde_json::to_string_pretty(&json!({
        "weights": {
            names[0]: weights[0],
            names[1]: weights[1],
            names[2]: weights[2],
            names[3]: weights[3],
        }
    }))
    .unwrap();
    std::fs::write(&out_path, &obj).expect("write weights");
    eprintln!("saved weights to {out_path}");
}

fn build_dev_cases(
    path: &str,
    decoder: &ModelDecoder,
    featurizer: &Reranker,
) -> Vec<DevCase> {
    let f = File::open(path).expect("open dev");
    let mut cases = Vec::new();
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let rec: Record = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let roman = rec.roman.to_ascii_lowercase();
        let target = rec.target.trim();
        if roman.is_empty() || target.is_empty() {
            continue;
        }
        let decoded = decoder.decode_detailed(&roman, 50);
        let cands: Vec<[f64; NUM_FEATURES]> = decoded
            .iter()
            .map(|c| featurizer.features(&roman, c))
            .collect();
        let target_pos = decoded.iter().position(|c| c.dev == target);
        cases.push(DevCase { cands, target_pos });
    }
    cases
}

fn evaluate(weights: &[f64; NUM_FEATURES], cases: &[DevCase]) -> usize {
    cases
        .iter()
        .filter(|case| {
            case.target_pos.map_or(false, |tp| {
                let mut best_i = 0usize;
                let mut best_s = f64::NEG_INFINITY;
                for (i, feat) in case.cands.iter().enumerate() {
                    let s: f64 = weights.iter().zip(feat.iter()).map(|(w, x)| w * x).sum();
                    if s > best_s {
                        best_s = s;
                        best_i = i;
                    }
                }
                best_i == tp
            })
        })
        .count()
}

fn next_value(flag: &str, val: Option<String>) -> String {
    val.unwrap_or_else(|| {
        eprintln!("Missing value for {flag}");
        std::process::exit(2);
    })
}
