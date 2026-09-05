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
    while let Some(a) = args.next() {
        if a == "--in" {
            input = args.next().expect("--in <path>");
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
            parts.next().unwrap_or(""),
            parts.next().unwrap_or(""),
            parts.next().unwrap_or("0").parse::<u32>().unwrap_or(0),
        );
        if w1.is_empty() || w2.is_empty() || freq == 0 {
            continue;
        }
        map.entry(w1.to_string())
            .or_default()
            .push((w2.to_string(), freq));
    }
    // Sort successors by frequency descending for compact, cache-friendly lookups.
    for succ in map.values_mut() {
        succ.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    }
    let bytes = bincode::serialize(&map).expect("serialize");
    let out = "data/word_bigrams.bin";
    let mut f = std::fs::File::create(out).expect("create");
    f.write_all(&bytes).expect("write");
    let n_pairs: usize = map.values().map(|v| v.len()).sum();
    eprintln!(
        "{n_pairs} bigram pairs over {} context words -> {out} ({:.1} MB)",
        map.len(),
        bytes.len() as f64 / 1e6
    );
}
