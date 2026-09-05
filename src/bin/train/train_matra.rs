// File: src/bin/train_matra.rs
//
// W4: Factored Matra Discriminative Model.
//
// 1. Deconstructs Devanagari words into consonant slots + matras.
// 2. Extracts aligned Roman vowel context:
//    - (Consonant x Roman vowel slice)
//    - (Matra ID x Roman vowel slice)
//    - Position (initial, medial, final)
// 3. Scores log P(V | C, R) for each candidate.
// 4. Measures test set reranking improvement on pure matra confusion pairs.

use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

#[allow(dead_code)]
const MATRAS: [char; 11] = [
    '\u{093E}', // 0: ा (aa)
    '\u{093F}', // 1: ि (i)
    '\u{0940}', // 2: ी (ii)
    '\u{0941}', // 3: ु (u)
    '\u{0942}', // 4: ू (uu)
    '\u{0947}', // 5: े (e)
    '\u{0948}', // 6: ै (ai)
    '\u{094B}', // 7: ो (o)
    '\u{094C}', // 8: ौ (au)
    '\u{0943}', // 9: ृ (ri)
    '\u{094D}', // 10: ् (halant / pure consonant)
];

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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MatraClass {
    None, // Bare consonant or implicit 'a'
    Aa,   // ा
    I,    // ि
    Ii,   // ी
    U,    // ु
    Uu,   // ू
    E,    // े
    Ai,   // ै
    O,    // ो
    Au,   // ौ
    Ri,   // ृ
    Hal,  // ्
}

impl MatraClass {
    pub fn from_char(ch: char) -> Option<Self> {
        match ch {
            '\u{093E}' => Some(Self::Aa),
            '\u{093F}' => Some(Self::I),
            '\u{0940}' => Some(Self::Ii),
            '\u{0941}' => Some(Self::U),
            '\u{0942}' => Some(Self::Uu),
            '\u{0947}' => Some(Self::E),
            '\u{0948}' => Some(Self::Ai),
            '\u{094B}' => Some(Self::O),
            '\u{094C}' => Some(Self::Au),
            '\u{0943}' => Some(Self::Ri),
            '\u{094D}' => Some(Self::Hal),
            _ => None,
        }
    }

    pub fn idx(&self) -> usize {
        match self {
            Self::None => 0,
            Self::Aa => 1,
            Self::I => 2,
            Self::Ii => 3,
            Self::U => 4,
            Self::Uu => 5,
            Self::E => 6,
            Self::Ai => 7,
            Self::O => 8,
            Self::Au => 9,
            Self::Ri => 10,
            Self::Hal => 11,
        }
    }
}

/// Slot extraction from Devanagari string
pub fn extract_slots(dev: &str) -> Vec<(char, MatraClass, bool)> {
    let mut slots = Vec::new();
    let chars: Vec<char> = dev.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        let ch = chars[i];
        // If it's a Devanagari consonant or independent vowel
        if ('\u{0915}'..='\u{0939}').contains(&ch) || ('\u{0905}'..='\u{0914}').contains(&ch) {
            let mut matra = MatraClass::None;
            if i + 1 < n {
                if let Some(m) = MatraClass::from_char(chars[i + 1]) {
                    matra = m;
                    i += 1;
                }
            }
            let is_final = i + 1 >= n;
            slots.push((ch, matra, is_final));
        }
        i += 1;
    }
    slots
}

/// Simple heuristic compatibility score between Roman string and Devanagari matras
pub fn matra_roman_log_prob(dev: &str, roman: &str) -> f64 {
    let slots = extract_slots(dev);
    let mut score = 0.0f64;
    let r_lower = roman.to_ascii_lowercase();

    for (_c, m, is_final) in &slots {
        match m {
            MatraClass::Aa => {
                if r_lower.contains("aa") {
                    score += 2.0;
                } else if r_lower.contains('a') {
                    score += 0.5;
                }
            }
            MatraClass::Ii => {
                if r_lower.contains("ee") {
                    score += 2.5;
                } else if *is_final {
                    score += 0.8; // Final vowels in Nepali overwhelmingly favor dirgha ी
                }
            }
            MatraClass::I => {
                if r_lower.contains("ee") {
                    score -= 1.5;
                } else if !*is_final && r_lower.contains('i') {
                    score += 0.5;
                }
            }
            MatraClass::Uu => {
                if r_lower.contains("oo") {
                    score += 2.5;
                }
            }
            MatraClass::U => {
                if r_lower.contains("oo") {
                    score -= 1.5;
                } else if r_lower.contains('u') {
                    score += 0.5;
                }
            }
            MatraClass::Ai => {
                if r_lower.contains("ai") {
                    score += 2.0;
                }
            }
            MatraClass::Au
                if r_lower.contains("au") => {
                    score += 2.0;
                }
            _ => {}
        }
    }
    score
}

fn main() {
    let t0 = Instant::now();
    eprintln!("============================================================");
    eprintln!("       W4: Factored Matra Discriminative Model Smoke Test   ");
    eprintln!("============================================================");

    let test_path = "/tmp/dump_test.jsonl";
    let valid_path = "/tmp/dump_valid.jsonl";
    if !Path::new(test_path).exists() {
        panic!("test dump not found");
    }

    // Load vocabulary
    let vocab_path = "data/word_freq_text.bin";
    let bytes = std::fs::read(vocab_path).expect("read vocab file");
    let freqs: HashMap<String, u32> = bincode::deserialize(&bytes).expect("deserialize vocab");

    let evaluate_matra_rescoring = |path: &str, desc: &str, gamma_matra: f64| {
        let file = File::open(path).expect("open dump");
        let mut total = 0usize;
        let mut base_hits = 0usize;
        let mut rescored_hits = 0usize;
        let mut matra_miss_recovered = 0usize;
        let mut matra_miss_hurt = 0usize;

        for line in BufReader::new(file).lines().map_while(Result::ok) {
            if let Ok(rec) = serde_json::from_str::<DumpRecord>(&line) {
                if rec.cands.is_empty() {
                    continue;
                }
                total += 1;

                // Base heuristic
                let mut best_base_s = f64::INFINITY;
                let mut best_base_d = "";
                for DumpCandidate(dev, emit, lm, _) in &rec.cands {
                    let f = freqs.get(dev).copied().unwrap_or(0);
                    let s = emit + 0.85 * lm - 0.75 * (1.0 + f as f64).ln();
                    if s < best_base_s {
                        best_base_s = s;
                        best_base_d = dev;
                    }
                }

                // Rescored with Factored Matra feature
                let mut best_rescore_s = f64::INFINITY;
                let mut best_rescore_d = "";
                for DumpCandidate(dev, emit, lm, _) in &rec.cands {
                    let f = freqs.get(dev).copied().unwrap_or(0);
                    let matra_score = matra_roman_log_prob(dev, &rec.roman);
                    // Higher matra_score reduces overall cost
                    let s = emit + 0.85 * lm - 0.75 * (1.0 + f as f64).ln() - gamma_matra * matra_score;
                    if s < best_rescore_s {
                        best_rescore_s = s;
                        best_rescore_d = dev;
                    }
                }

                let base_ok = best_base_d == rec.gold;
                let rescored_ok = best_rescore_d == rec.gold;

                if base_ok {
                    base_hits += 1;
                }
                if rescored_ok {
                    rescored_hits += 1;
                }
                if !base_ok && rescored_ok {
                    matra_miss_recovered += 1;
                }
                if base_ok && !rescored_ok {
                    matra_miss_hurt += 1;
                }
            }
        }

        let base_acc = (base_hits as f64 / total as f64) * 100.0;
        let rescore_acc = (rescored_hits as f64 / total as f64) * 100.0;
        eprintln!(
            "{} (gamma={:.2}): Base = {:.2}% -> Matra Rescored = {:.2}% ({:+.2}%) | Recovered: +{}, Hurt: -{}, Net: {:+}",
            desc, gamma_matra, base_acc, rescore_acc, rescore_acc - base_acc, matra_miss_recovered, matra_miss_hurt, matra_miss_recovered as i64 - matra_miss_hurt as i64
        );
    };

    eprintln!("--- Sweep of Matra Weight on Valid Set ---");
    for &gamma in &[0.05, 0.10, 0.20, 0.30, 0.50] {
        evaluate_matra_rescoring(valid_path, "Valid Set", gamma);
    }

    eprintln!("\n--- Evaluation with Optimal Matra Weight on Test Set ---");
    evaluate_matra_rescoring(test_path, "Test Set ", 0.20);

    eprintln!("\nCompleted in {:.2}s", t0.elapsed().as_secs_f64());
}
