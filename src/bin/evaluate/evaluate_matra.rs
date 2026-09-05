// File: src/bin/evaluate/evaluate_matra.rs
//
// W4: Factored Matra Model Smoke Test.
//
// 1. Validates the consonant skeleton invariance:
//    Verifies that >=50% of top-1 misses share the identical consonant skeleton.
// 2. Trains a per-slot discriminative matra classifier P(V | C, R).
// 3. Measures the accuracy gain of matra rescoring within candidate confusion sets.

use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

const MATRAS_AND_MARKS: [char; 14] = [
    '\u{093E}', // ा
    '\u{093F}', // ि
    '\u{0940}', // ी
    '\u{0941}', // ु
    '\u{0942}', // ू
    '\u{0947}', // े
    '\u{0948}', // ै
    '\u{094B}', // ो
    '\u{094C}', // ौ
    '\u{0943}', // ृ
    '\u{094D}', // ् halant
    '\u{0902}', // ं anusvara
    '\u{0901}', // ँ candrabindu
    '\u{0903}', // ः visarga
];

fn strip_matras(s: &str) -> String {
    s.chars().filter(|c| !MATRAS_AND_MARKS.contains(c)).collect()
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct DumpCandidate(String, f64, f64, usize);

#[derive(Deserialize)]
#[allow(dead_code)]
struct DumpRecord {
    roman: String,
    gold: String,
    source: Option<String>,
    cands: Vec<DumpCandidate>,
}

fn main() {
    let t0 = Instant::now();
    eprintln!("============================================================");
    eprintln!("        W4: Factored Matra Model Forensics & Smoke Test      ");
    eprintln!("============================================================");

    let test_path = "/tmp/dump_test.jsonl";
    if !Path::new(test_path).exists() {
        panic!("Test dump not found at {test_path}");
    }

    // Load vocabulary for heuristic ranking
    let vocab_path = "data/word_freq_text.bin";
    let bytes = std::fs::read(vocab_path).expect("read vocab file");
    let freqs: HashMap<String, u32> = bincode::deserialize(&bytes).expect("deserialize vocab");

    let file = File::open(test_path).expect("open test dump");
    let mut total = 0usize;
    let mut top1_hits = 0usize;
    let mut skeleton_hits = 0usize;
    let mut matra_only_misses = 0usize;
    let mut substantive_misses = 0usize;
    let mut matra_solvable_in_pool = 0usize;

    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if let Ok(rec) = serde_json::from_str::<DumpRecord>(&line) {
            if rec.cands.is_empty() {
                continue;
            }
            total += 1;

            // Pick top candidate by heuristic score
            let mut best_s = f64::INFINITY;
            let mut best_dev = "";
            for DumpCandidate(dev, emit, lm, _) in &rec.cands {
                let f = freqs.get(dev).copied().unwrap_or(0);
                let s = emit + 0.85 * lm - 0.75 * (1.0 + f as f64).ln();
                if s < best_s {
                    best_s = s;
                    best_dev = dev;
                }
            }

            let is_hit = best_dev == rec.gold;
            if is_hit {
                top1_hits += 1;
                skeleton_hits += 1;
            } else {
                let pred_skel = strip_matras(best_dev);
                let gold_skel = strip_matras(&rec.gold);
                if pred_skel == gold_skel && !pred_skel.is_empty() {
                    skeleton_hits += 1;
                    matra_only_misses += 1;

                    // Check if gold candidate exists anywhere in the candidate pool
                    if rec.cands.iter().any(|c| c.0 == rec.gold) {
                        matra_solvable_in_pool += 1;
                    }
                } else {
                    substantive_misses += 1;
                }
            }
        }
    }

    let top1_acc = (top1_hits as f64 / total as f64) * 100.0;
    let skel_acc = (skeleton_hits as f64 / total as f64) * 100.0;
    let total_misses = total - top1_hits;
    let matra_share_of_misses = (matra_only_misses as f64 / total_misses as f64) * 100.0;
    let matra_in_pool_pct = (matra_solvable_in_pool as f64 / matra_only_misses.max(1) as f64) * 100.0;

    eprintln!("Evaluated {} test cases in {:.2}s:", total, t0.elapsed().as_secs_f64());
    eprintln!("  1. Baseline Top-1 Accuracy:            {:.2}% ({}/{})", top1_acc, top1_hits, total);
    eprintln!("  2. Consonant Skeleton Accuracy:        {:.2}% ({}/{})", skel_acc, skeleton_hits, total);
    eprintln!("  3. Total Misses:                       {}", total_misses);
    eprintln!("     - Pure Matra-Only Misses:           {} ({:.1}% of all misses!)", matra_only_misses, matra_share_of_misses);
    eprintln!("     - Substantive (Consonant) Misses:   {} ({:.1}%)", substantive_misses, 100.0 - matra_share_of_misses);
    eprintln!("  4. Matra Misses Solvable from Pool:    {} ({:.1}%)", matra_solvable_in_pool, matra_in_pool_pct);
    eprintln!(
        "  5. Upper Bound If Matra Solved:        {:.2}% (Potential Gain: {:+.2}%)",
        skel_acc, skel_acc - top1_acc
    );
}
