// File: src/bin/build_bigrams.rs
//
// E6: serialize the corpus word-bigram table (data/store/word_pairs.csv,
// w1,w2,freq) into a compact binary artifact the engine loads for
// context-conditioned suggestion reranking.
//
// Usage: cargo run --release --bin build_bigrams  [--in data/store/word_pairs.csv]
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut input = "data/store/word_pairs.csv".to_string();
    let mut out = "data/word_bigrams.bin".to_string();
    let mut min_freq: u32 = 1;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--in" => input = args.next().expect("--in <path>"),
            "--out" => out = args.next().expect("--out <path>"),
            "--min-freq" => min_freq = args.next().expect("--min-freq <n>").parse().unwrap(),
            "-h" | "--help" => {
                println!("Usage: build_bigrams [--in path] [--out path] [--min-freq n]");
                return;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }
    let mut map: HashMap<String, Vec<(String, u32)>> = HashMap::new();
    let file = std::fs::File::open(&input).unwrap_or_else(|e| panic!("open {input}: {e}"));
    for (i, line) in BufReader::new(file).lines().enumerate() {
        let line = line.expect("read");
        if i == 0 || line.trim().is_empty() {
            continue; // header
        }
        let mut parts = line.splitn(3, ',');
        let (w1, w2, freq) = (
            parts.next().unwrap_or("").trim(),
            parts.next().unwrap_or("").trim(),
            parts.next().unwrap_or("0").trim().parse::<u32>().unwrap_or(0),
        );
        if !is_devanagari_word(w1) || !is_devanagari_word(w2) || freq < min_freq {
            continue;
        }
        map.entry(w1.to_string())
            .or_default()
            .push((w2.to_string(), freq));
    }
    // Sort successors by frequency descending for compact, cache-friendly lookups.
    for succ in map.values_mut() {
        succ.sort_unstable_by_key(|b| std::cmp::Reverse(b.1));
    }
    let bytes = bincode::serialize(&map).expect("serialize");
    let mut f = std::fs::File::create(&out).expect("create");
    f.write_all(&bytes).expect("write");
    let n_pairs: usize = map.values().map(|v| v.len()).sum();
    eprintln!(
        "{n_pairs} bigram pairs over {} context words -> {out} ({:.1} MB)",
        map.len(),
        bytes.len() as f64 / 1e6
    );
}

fn is_devanagari_word(w: &str) -> bool {
    let count = w.chars().count();
    count >= 1 && count <= 24 && w.chars().all(|c| ('\u{0900}'..='\u{0963}').contains(&c))
}
