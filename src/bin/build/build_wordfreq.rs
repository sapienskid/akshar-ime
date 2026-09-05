// File: src/bin/build_wordfreq.rs
//
// E3: build a Devanagari word-frequency map from the native side of the
// Aksharantar corpus.  Serialised as bincode HashMap<String, u32>.
//
// Usage: cargo run --release --bin build_wordfreq -- \
//          data/aksharantar/nep_train.json [more.json ...]

use serde::Deserialize;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(Deserialize)]
struct Record<'a> {
    #[serde(rename = "native word")]
    native: &'a str,
}

fn main() {
    let paths: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if paths.is_empty() {
        eprintln!("usage: build_wordfreq <jsonl...>  (output: data/word_freq.bin)");
        std::process::exit(2);
    }
    let mut freq: HashMap<String, u32> = HashMap::new();
    for path in &paths {
        let Ok(f) = std::fs::File::open(path) else {
            eprintln!("WARNING: cannot open {}", path.display());
            continue;
        };
        for line in BufReader::new(f).lines().map_while(Result::ok) {
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            if let Ok(rec) = serde_json::from_str::<Record>(t) {
                let native = rec.native.trim();
                if !native.is_empty() {
                    *freq.entry(native.to_string()).or_insert(0) += 1;
                }
            }
        }
    }
    let out = std::path::Path::new("data/word_freq.bin");
    let bytes = bincode::serialize(&freq).expect("serialize");
    std::fs::write(out, &bytes).expect("write");
    eprintln!(
        "{} unique words, total {} occurrences -> {} ({:.1} MB)",
        freq.len(),
        freq.values().sum::<u32>(),
        out.display(),
        bytes.len() as f64 / 1e6
    );
}
