// File: src/bin/build_wordfreq_text.rs
//
// M4-real: count Devanagari word frequencies from raw running text
// (e.g. the Nepali Wikipedia dump) and serialise
// HashMap<String, u32> -> data/word_freq_text.bin.
//
// Usage: build_wordfreq_text <text-file> [more files...]

use std::collections::HashMap;
use std::io::{BufRead, BufReader};

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: build_wordfreq_text <raw-text...>");
        std::process::exit(2);
    }
    let mut freq: HashMap<String, u32> = HashMap::new();
    let mut total = 0u64;
    for path in &paths {
        let Ok(f) = std::fs::File::open(path) else {
            eprintln!("WARNING: cannot open {path}");
            continue;
        };
        for line in BufReader::new(f).lines().map_while(Result::ok) {
            for word in line.split_whitespace() {
                // Keep tokens that are (mostly) Devanagari letters.
                let chars: Vec<char> = word.chars().collect();
                if chars.is_empty() || chars.len() > 24 {
                    continue;
                }
                let dev = chars
                    .iter()
                    .filter(|c| matches!(c, '\u{0900}'..='\u{097F}'))
                    .count();
                if dev * 2 >= chars.len() {
                    *freq.entry(word.to_string()).or_insert(0) += 1;
                    total += 1;
                }
            }
        }
    }
    // Prune hapax-ish tail to keep the artifact sane.
    freq.retain(|_, &mut c| c >= 3);
    let out = std::path::Path::new("data/word_freq_text.bin");
    let bytes = bincode::serialize(&freq).expect("serialize");
    std::fs::write(out, &bytes).expect("write");
    eprintln!(
        "{total} tokens -> {} words (count>=3) -> {} ({:.1} MB)",
        freq.len(),
        out.display(),
        bytes.len() as f64 / 1e6
    );
}
