// File: src/bin/build_lexicon.rs
//
// Build the Roman -> Devanagari word lexicon from the Aksharantar Nepali
// corpus and serialise it to `data/roman_lexicon.bin`.
//
// Usage:
//   cargo run --release --bin build_lexicon -- [options]
//     --train <path>   train split (default: data/aksharantar/nep_train.json)
//     --extra <path>   extra split merged in (e.g. nep_valid.json)
//     --out <path>     output lexicon path (default: data/roman_lexicon.bin)

use akshar_ime::core::lexicon::RomanLexicon;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Deserialize)]
struct Record<'a> {
    #[serde(rename = "english word")]
    english: &'a str,
    #[serde(rename = "native word")]
    native: &'a str,
}

fn main() {
    let mut train = PathBuf::from("data/aksharantar/nep_train.json");
    let mut extra: Option<PathBuf> = None;
    let mut out = PathBuf::from("data/roman_lexicon.bin");

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--train" => train = PathBuf::from(next_value(&arg, args.next())),
            "--extra" => extra = Some(PathBuf::from(next_value(&arg, args.next()))),
            "--out" => out = PathBuf::from(next_value(&arg, args.next())),
            "--help" | "-h" => {
                println!("usage: build_lexicon [--train path] [--extra path] [--out path]");
                return;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }

    let mut pairs: Vec<(String, String)> = Vec::new();
    let t = Instant::now();
    ingest(&train, &mut pairs);
    if let Some(extra) = &extra {
        ingest(extra, &mut pairs);
    }
    eprintln!("Collected {} pairs in {:.1}s", pairs.len(), t.elapsed().as_secs_f64());

    let lexicon = RomanLexicon::build(pairs);
    eprintln!("Lexicon entries: {}", lexicon.len());

    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    let t = Instant::now();
    lexicon.save(&out).expect("failed to save lexicon");
    let size = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "Saved lexicon to {} ({:.1} MB) in {:.1}s",
        out.display(),
        size as f64 / 1e6,
        t.elapsed().as_secs_f64()
    );
}

fn ingest(path: &PathBuf, pairs: &mut Vec<(String, String)>) {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("WARNING: could not open {}: {e}", path.display());
            return;
        }
    };
    let reader = BufReader::new(file);
    let mut count = 0usize;
    for line in reader.lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let rec: Record = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(_) => continue,
        };
        pairs.push((rec.english.to_string(), rec.native.to_string()));
        count += 1;
    }
    eprintln!("  [{}] {count} pairs", path.display());
}

fn next_value(flag: &str, val: Option<String>) -> String {
    val.unwrap_or_else(|| {
        eprintln!("Missing value for {flag}");
        std::process::exit(2);
    })
}
