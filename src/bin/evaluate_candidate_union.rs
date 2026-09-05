// File: src/bin/smoke_w3_union.rs
//
// W3: Candidate Union Smoke Test.
// Measures oracle top-K coverage of:
//   (a) Base beam candidates alone
//   (b) Base + Word-Trie candidates
//   (c) Multi-generator union

use serde::Deserialize;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Deserialize)]
#[allow(dead_code)]
struct DumpCandidate(String, f64, f64, usize);

#[derive(Deserialize)]
#[allow(dead_code)]
struct TrieCandidate(String, f64, u32);

#[derive(Deserialize)]
#[allow(dead_code)]
struct DumpRecord {
    roman: String,
    gold: String,
    source: Option<String>,
    cands: Vec<DumpCandidate>,
    trie: Option<Vec<TrieCandidate>>,
}

fn evaluate_oracles(path: &str, name: &str) {
    let file = File::open(path).expect("open dump");
    let mut total = 0usize;
    let mut base_oracle = 0usize;
    let mut union_oracle = 0usize;
    let mut base_top1 = 0usize;

    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if let Ok(rec) = serde_json::from_str::<DumpRecord>(&line) {
            total += 1;
            let mut seen: HashSet<String> = HashSet::new();

            let in_base = rec.cands.iter().any(|c| c.0 == rec.gold);
            if in_base {
                base_oracle += 1;
            }
            if rec.cands.first().is_some_and(|c| c.0 == rec.gold) {
                base_top1 += 1;
            }

            for c in &rec.cands {
                seen.insert(c.0.clone());
            }
            if let Some(trie) = &rec.trie {
                for t in trie {
                    seen.insert(t.0.clone());
                }
            }

            if seen.contains(&rec.gold) {
                union_oracle += 1;
            }
        }
    }

    let base_t1_pct = (base_top1 as f64 / total as f64) * 100.0;
    let base_ora_pct = (base_oracle as f64 / total as f64) * 100.0;
    let union_ora_pct = (union_oracle as f64 / total as f64) * 100.0;

    eprintln!(
        "{} ({} cases):\n  Base Top-1: {:.2}%\n  Base Oracle (Top-50): {:.2}%\n  Base + Trie Union Oracle: {:.2}% ({:+.2}% gain, +{} words unlocked)",
        name, total, base_t1_pct, base_ora_pct, union_ora_pct, union_ora_pct - base_ora_pct, union_oracle - base_oracle
    );
}

fn main() {
    eprintln!("============================================================");
    eprintln!("            W3: Candidate Union Oracle Smoke Test           ");
    eprintln!("============================================================");

    if Path::new("/tmp/dump_valid.jsonl").exists() {
        evaluate_oracles("/tmp/dump_valid.jsonl", "Valid Split");
    }
    if Path::new("/tmp/dump_test.jsonl").exists() {
        evaluate_oracles("/tmp/dump_test.jsonl", "Test Split ");
    }
}
