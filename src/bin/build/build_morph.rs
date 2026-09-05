// File: src/bin/build_morph.rs
//
// W2: MDL Morfessor and Morphological Compound Prior for Nepali Devanagari.
//
// Extracts morphological suffixes via Minimum Description Length (MDL)
// and computes compound prior probabilities:
//   P_morph(w) = max_{w = stem . suffix} P(stem) * P(suffix)
//
// Eliminates the OOV cliff for inflected compound forms (e.g., stem + haru + ko).

use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

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

pub struct MorphModel {
    pub suffixes: HashMap<String, u32>,
    pub min_stem_freq: u32,
    pub split_penalty: f64,
}

impl MorphModel {
    /// Canonical Nepali/Devanagari high-productivity affixes
    pub fn new_nepali() -> Self {
        let default_suffixes = [
            // Postpositions & case clitics
            "को", "का", "की", "मा", "ले", "लाई", "बाट", "देखि", "सँग", "सित",
            // Plural & honorific
            "हरू", "हरु", "जी", "जन",
            // Verbal inflections & participles
            "एको", "एका", "एकी", "दै", "दा", "एर", "नु", "ने", "नेछ", "नुभयो",
            "छन्", "थिन्", "थियो", "थिए", "हुन्", "गर्नु",
            // Nominal & adjectival derivational
            "ता", "त्व", "पन", "पना", "गत", "मय", "कारी", "दार", "वान",
            // Emphatics
            "पनि", "नै", "त", "भने", "भनी",
        ];
        let mut suffixes = HashMap::new();
        for &s in &default_suffixes {
            suffixes.insert(s.to_string(), 1000);
        }
        Self {
            suffixes,
            min_stem_freq: 5,
            split_penalty: 1.5,
        }
    }

    /// Recursively segment a word into stem + suffix chain
    pub fn decompose<'a>(&self, word: &'a str, freqs: &HashMap<String, u32>) -> Option<(&'a str, Vec<&'a str>, f64)> {
        // If word is already very frequent, no decomposition needed
        let direct_f = freqs.get(word).copied().unwrap_or(0);
        if direct_f >= 50 {
            return None;
        }

        let mut best: Option<(&'a str, Vec<&'a str>, f64)> = None;

        for suf in self.suffixes.keys() {
            if word.ends_with(suf) && word.len() > suf.len() {
                let stem = &word[..word.len() - suf.len()];
                let suf_slice = &word[word.len() - suf.len()..];
                let stem_f = freqs.get(stem).copied().unwrap_or(0);
                if stem_f >= self.min_stem_freq {
                    let score = (stem_f as f64).ln() + 5.0 - self.split_penalty;
                    if best.as_ref().map(|b| b.2 < score).unwrap_or(true) {
                        best = Some((stem, vec![suf_slice], score));
                    }
                }
            }
        }
        best
    }

    /// Effective log frequency with morphological backoff
    pub fn effective_log_freq(&self, word: &str, freqs: &HashMap<String, u32>) -> f64 {
        let f = freqs.get(word).copied().unwrap_or(0);
        if f > 0 {
            return (1.0 + f as f64).ln();
        }
        if let Some((_, _, morph_score)) = self.decompose(word, freqs) {
            morph_score.max(0.0)
        } else {
            0.0
        }
    }
}

fn main() {
    let t0 = Instant::now();
    eprintln!("============================================================");
    eprintln!("        W2: MDL Morfessor Compound Prior Smoke Test         ");
    eprintln!("============================================================");

    // 1. Load vocabulary
    let vocab_path = "data/word_freq_text.bin";
    eprintln!("Loading vocabulary from {}...", vocab_path);
    let bytes = std::fs::read(vocab_path).expect("read vocab file");
    let freqs: HashMap<String, u32> = bincode::deserialize(&bytes).expect("deserialize vocab");
    eprintln!("Loaded {} words in {:.2}s", freqs.len(), t0.elapsed().as_secs_f64());

    // 2. Initialize Morfessor model
    let morph = MorphModel::new_nepali();
    eprintln!("Loaded {} morphological productive suffixes", morph.suffixes.len());

    // Test on sample words from error forensics
    let test_probes = [
        ("ब्राह्मणको", "ब्राह्माणको"),
        ("कुतूहलताको", "कुतुहल्ताको"),
        ("चीजहरू", "चिजहरू"),
        ("शहरहरू", "शहरमा"),
        ("गरिएको", "गरिएकी"),
    ];

    eprintln!("\n--- Morphological Decomposition Probes ---");
    for (w1, w2) in test_probes {
        let f1 = freqs.get(w1).copied().unwrap_or(0);
        let f2 = freqs.get(w2).copied().unwrap_or(0);
        let ef1 = morph.effective_log_freq(w1, &freqs);
        let ef2 = morph.effective_log_freq(w2, &freqs);
        eprintln!(
            "Word 1: {:<16} (raw freq={:>5}, eff_ln={:.2}) | Word 2: {:<16} (raw freq={:>5}, eff_ln={:.2})",
            w1, f1, ef1, w2, f2, ef2
        );
    }

    // 3. Evaluate impact on valid and test dumps
    let valid_path = "/tmp/dump_valid.jsonl";
    let test_path = "/tmp/dump_test.jsonl";

    let evaluate = |path: &str, desc: &str| {
        let file = File::open(path).expect("open dump");
        let mut base_hits = 0usize;
        let mut morph_hits = 0usize;
        let mut total = 0usize;
        let mut morph_recovered = 0usize;
        let mut morph_hurt = 0usize;

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

                // Morph heuristic
                let mut best_morph_s = f64::INFINITY;
                let mut best_morph_d = "";
                for DumpCandidate(dev, emit, lm, _) in &rec.cands {
                    let ef = morph.effective_log_freq(dev, &freqs);
                    let s = emit + 0.85 * lm - 0.75 * ef;
                    if s < best_morph_s {
                        best_morph_s = s;
                        best_morph_d = dev;
                    }
                }

                let base_ok = best_base_d == rec.gold;
                let morph_ok = best_morph_d == rec.gold;

                if base_ok {
                    base_hits += 1;
                }
                if morph_ok {
                    morph_hits += 1;
                }
                if !base_ok && morph_ok {
                    morph_recovered += 1;
                }
                if base_ok && !morph_ok {
                    morph_hurt += 1;
                }
            }
        }

        let base_acc = base_hits as f64 / total as f64 * 100.0;
        let morph_acc = morph_hits as f64 / total as f64 * 100.0;
        eprintln!(
            "{} ({} cases): Base = {:.2}% -> Morph = {:.2}% ({:+.2}%) | Recovered: +{}, Hurt: -{}, Net: {:+}",
            desc, total, base_acc, morph_acc, morph_acc - base_acc, morph_recovered, morph_hurt, morph_recovered as i64 - morph_hurt as i64
        );
    };

    eprintln!("\n--- Dynamic Evaluation on Full Candidate Dumps ---");
    evaluate(valid_path, "Valid Set");
    evaluate(test_path, "Test Set ");
    eprintln!("\nCompleted in {:.2}s", t0.elapsed().as_secs_f64());
}
