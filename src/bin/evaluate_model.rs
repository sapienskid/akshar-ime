// File: src/bin/evaluate_model.rs
//
// Benchmark: score the generative decoder against the held-out
// Aksharantar Nepali test split.  Compares directly with the IndicXlit
// reference numbers:
//
//   native words top-1: 80.25%     named entities top-1: 52.67%
//
// Usage:
//   cargo run --release --bin evaluate_model -- [options]
//     --model <path>    translit model (default: data/translit_model.bin)
//     --dataset <path>  JSONL test split (default: data/aksharantar/nep_test.json)
//     --topk <n>        top-k for hit metric (default: 5)
//     --show-misses <n> number of misses to print (default: 12)

use akshar_ime::core::translit_model::TranslitModel;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

#[derive(Deserialize)]
struct Record<'a> {
    #[serde(rename = "english word")]
    roman: &'a str,
    #[serde(rename = "native word")]
    target: &'a str,
    source: &'a str,
}

#[derive(Debug, Clone)]
struct EvalCase {
    roman: String,
    target: String,
    source: String,
}

#[derive(Debug, Clone, Default)]
struct Stats {
    total: usize,
    top1: usize,
    topk: usize,
}

impl Stats {
    fn add(&mut self, top1_hit: bool, topk_hit: bool) {
        self.total += 1;
        if top1_hit {
            self.top1 += 1;
        }
        if topk_hit {
            self.topk += 1;
        }
    }
}

fn main() {
    let mut model_path = "data/translit_model.bin".to_string();
    let mut dataset_path = "data/aksharantar/nep_test.json".to_string();
    let mut topk = 5usize;
    let mut show_misses = 12usize;
    let mut lm_weight = 1.0f64;
    let mut beam = 64usize;
    let mut per_chunk = 16usize;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_path = next_value(&arg, args.next()),
            "--dataset" => dataset_path = next_value(&arg, args.next()),
            "--topk" => topk = next_value(&arg, args.next()).parse::<usize>().expect("--topk <n>").max(1),
            "--show-misses" => show_misses = next_value(&arg, args.next()).parse().expect("--show-misses <n>"),
            "--lm-weight" => lm_weight = next_value(&arg, args.next()).parse().expect("--lm-weight <f>"),
            "--beam" => beam = next_value(&arg, args.next()).parse::<usize>().expect("--beam <n>").max(1),
            "--per-chunk" => per_chunk = next_value(&arg, args.next()).parse::<usize>().expect("--per-chunk <n>").max(1),
            "--help" | "-h" => {
                print_help();
                return;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                print_help();
                std::process::exit(2);
            }
        }
    }

    let model = TranslitModel::load(Path::new(&model_path))
        .unwrap_or_else(|e| panic!("load model {model_path}: {e}"));
    assert!(model.validate(), "model failed validation");
    let config = akshar_ime::core::decoder::DecoderConfig {
        beam_width: beam,
        max_aksharas_per_chunk: per_chunk,
        lm_weight,
        ..Default::default()
    };
    let decoder = akshar_ime::core::decoder::ModelDecoder::with_config(model, config);
    eprintln!(
        "Decoder ready (aksharas={}, chunks={}, lm_weight={lm_weight}, beam={beam})",
        decoder.model.aksharas.len(),
        decoder.model.chunks.len()
    );

    let cases = load_cases(&dataset_path);
    let mut by_source: HashMap<String, Stats> = HashMap::new();
    let mut total_stats = Stats::default();
    let mut misses: Vec<(String, String, Vec<String>)> = Vec::new();
    let mut decode_time = 0.0f64;

    for case in &cases {
        let t = Instant::now();
        let scored = decoder.decode(&case.roman, topk.max(8));
        decode_time += t.elapsed().as_secs_f64();
        let top: Vec<String> = scored.into_iter().map(|(d, _)| d).collect();
        let top1_hit = top.first().is_some_and(|d| d == &case.target);
        let topk_hit = top.iter().take(topk).any(|d| d == &case.target);
        by_source.entry(case.source.clone()).or_default().add(top1_hit, topk_hit);
        total_stats.add(top1_hit, topk_hit);
        if !topk_hit {
            misses.push((case.roman.clone(), case.target.clone(), top));
        }
    }

    println!("Generative decoder on Aksharantar Nepali test: {} cases", cases.len());
    println!("IndicXlit (neural, top-1) reference: native=80.25%, named-entities=52.67%");
    println!("Average decode time per word: {:.2} ms", decode_time / cases.len() as f64 * 1000.0);
    print_stats("ALL", &total_stats, topk);
    let mut sources: Vec<(&String, &Stats)> = by_source.iter().collect();
    sources.sort_by_key(|(k, _)| *k);
    for (name, s) in sources {
        print_stats(name, s, topk);
    }

    if show_misses > 0 {
        println!("\nMisses (first {show_misses}):");
        for (roman, target, top) in misses.iter().take(show_misses) {
            println!("  roman=`{roman}` target=`{target}` top={top:?}");
        }
    }
}

fn print_stats(name: &str, s: &Stats, topk: usize) {
    if s.total == 0 {
        return;
    }
    let t1 = s.top1 as f64 / s.total as f64 * 100.0;
    let tk = s.topk as f64 / s.total as f64 * 100.0;
    println!(
        "  {:<18} top1={:>5.2}%  top{topk}={:>5.2}%  ({}/{})",
        name, t1, tk, s.topk, s.total
    );
}

fn load_cases(path: &str) -> Vec<EvalCase> {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let reader = BufReader::new(file);
    let mut cases = Vec::new();
    for line in reader.lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let rec: Record = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let roman = rec.roman.trim().to_ascii_lowercase();
        let target = rec.target.trim().to_string();
        if roman.is_empty() || target.is_empty() {
            continue;
        }
        cases.push(EvalCase { roman, target, source: rec.source.to_string() });
    }
    cases
}

fn next_value(flag: &str, val: Option<String>) -> String {
    val.unwrap_or_else(|| {
        eprintln!("Missing value for {flag}");
        std::process::exit(2);
    })
}

fn print_help() {
    println!("Usage: cargo run --release --bin evaluate_model -- [options]");
    println!("  --model <path>    translit model (default: data/translit_model.bin)");
    println!("  --lm-weight <f>    LM weight relative to emissions (default: 1.0)");
    println!("  --dataset <path>  JSONL test split");
    println!("  --topk <n>        top-k for hit metric (default: 5)");
    println!("  --show-misses <n> misses to print (default: 12)");
    println!("  -h, --help        show help");
}
