//! Accuracy regression guard.
//!
//! A 30.8-point top-1 regression once shipped through a fully green unit-test
//! suite (the corpus-SymSpell candidate source scored above the decoder band,
//! displacing the reranker's top-1 on every query that produced a fuzzy hit).
//! Unit tests cannot catch that class of defect: every component was individually
//! correct and it was their *composition* that was wrong.  This test measures
//! end-to-end accuracy on a fixed sample of the held-out Aksharantar test
//! split and fails if it drops.
//!
//! Reads the vendored Nepali split (`data/aksharantar/nep_test.json`), which is
//! entirely AK-Freq, and whose records carry no `source` field -- see
//! `data/README.md`. The older `test_devanagari.jsonl` path is still accepted
//! when present, for a `data/` populated from an upstream dump.
//!
//! The sample is an even **stride** over the split, not a prefix: the AK-Freq
//! file is ordered, with the first ~2,050 cases averaging 8.7 codepoints and the
//! remainder 6.4, so a prefix sample reads ~22pp high (82.50% top-1 on the
//! first 400, against 60.35% on all 4,101). Striding brings the sample to
//! 58.75% / 76.00%, within sampling noise of the full-split figures, and costs
//! nothing.
//!
//! Thresholds are set below the measured baseline (2026-10-04, Nepali-only
//! container: top-1 60.35%, top-3 73.15% on the full 4,101-case split) with
//! room for sampling noise, so this catches regressions without failing on
//! ordinary variation. Raise them when a change genuinely improves the model.
//!
//! Skips (rather than fails) when the model or dataset is absent, so a fresh
//! clone without `data/` still passes CI.

use akshar_ime::ImeEngine;
use std::io::BufRead;
use std::path::Path;

const SAMPLE: usize = 400;

/// Measured 60.35% on the full AK-Freq split; allow for sampling noise.
const MIN_TOP1: f64 = 56.0;
/// Measured 73.15% on the full AK-Freq split (top-3 of the k=8 list, which is
/// the tightest of the three figures the eval harness reports).
const MIN_TOP5: f64 = 66.0;

/// The first split that exists, in preference order.
fn dataset() -> Option<&'static str> {
    [
        "data/aksharantar/nep_test.json",
        "data/aksharantar/test_devanagari.jsonl",
        "data/pairs/test.jsonl",
    ]
    .into_iter()
    .find(|p| Path::new(p).exists())
}

/// Every `stride`-th usable row, so the sample spans the whole ordered split.
fn load_native_pairs(path: &str, limit: usize) -> Vec<(String, String)> {
    let Ok(file) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let mut all = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        let Ok(line) = line else { break };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        // The cleaned Nepali split has no `source`; an upstream dump does, and
        // AK-Freq is the native-word stratum IndicXlit reports on.
        match v["source"].as_str() {
            Some(s) if s != "AK-Freq" => continue,
            _ => {}
        }
        if let (Some(r), Some(d)) = (v["english word"].as_str(), v["native word"].as_str()) {
            all.push((r.to_string(), d.to_string()));
        }
    }
    if all.len() <= limit {
        return all;
    }
    let stride = all.len() / limit;
    all.into_iter().step_by(stride).take(limit).collect()
}

#[test]
fn ak_freq_top1_and_top5_do_not_regress() {
    if !Path::new("data/akshar.model").exists() {
        eprintln!("skipping: data/akshar.model not present");
        return;
    }
    let Some(path) = dataset() else {
        eprintln!("skipping: no Aksharantar test split under data/");
        return;
    };
    let pairs = load_native_pairs(path, SAMPLE);
    if pairs.is_empty() {
        eprintln!("skipping: {path} yielded no usable rows");
        return;
    }

    let engine = ImeEngine::new();
    let (mut top1, mut top5) = (0usize, 0usize);
    for (roman, gold) in &pairs {
        let suggestions = engine.get_suggestions(roman, 5);
        if suggestions.first().is_some_and(|(s, _)| s == gold) {
            top1 += 1;
        }
        if suggestions.iter().any(|(s, _)| s == gold) {
            top5 += 1;
        }
    }

    let n = pairs.len() as f64;
    let p1 = top1 as f64 / n * 100.0;
    let p5 = top5 as f64 / n * 100.0;
    println!(
        "AK-Freq sample n={}: top-1 {:.2}%  top-5 {:.2}%",
        pairs.len(),
        p1,
        p5
    );

    assert!(
        p1 >= MIN_TOP1,
        "AK-Freq top-1 regressed to {p1:.2}% (floor {MIN_TOP1}%, n={}). \
         A candidate source is most likely outranking the decoder+reranker; \
         check the score bands in engine.rs.",
        pairs.len()
    );
    assert!(
        p5 >= MIN_TOP5,
        "AK-Freq top-5 regressed to {p5:.2}% (floor {MIN_TOP5}%, n={})",
        pairs.len()
    );
}

#[test]
#[ignore]
fn diagnose_unreachable_cases() {
    let Some(path) = dataset() else { return };
    let Ok(file) = std::fs::File::open(path) else {
        return;
    };
    let mut pairs = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        let Ok(line) = line else { continue };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        let roman = v
            .get("english word")
            .or_else(|| v.get("roman"))
            .and_then(|x| x.as_str());
        let gold = v
            .get("native word")
            .or_else(|| v.get("devanagari"))
            .and_then(|x| x.as_str());
        if let (Some(r), Some(g)) = (roman, gold) {
            pairs.push((r.to_string(), g.to_string()));
        }
    }
    if pairs.is_empty() {
        return;
    };

    let engine = ImeEngine::new();
    let total = pairs.len();
    let mut unreachable = 0usize;
    let mut oov_count = 0usize;
    let mut key_exact_count = 0usize;
    let mut key_dist_1 = 0usize;
    let mut key_dist_2 = 0usize;
    let mut key_dist_3plus = 0usize;
    let mut sample_unreachable = Vec::new();

    for (roman, gold) in &pairs {
        let suggestions = engine.get_suggestions(roman, 50);
        let reached = suggestions.iter().any(|(s, _)| s == gold);
        if !reached {
            unreachable += 1;
            // Check if gold in lexicon
            let in_lex = if let Ok(data) = std::fs::read("data/lexicon.bin") {
                if let Ok(ld) =
                    bincode::deserialize::<akshar_ime::core::lexicon::LexiconData>(&data)
                {
                    if let Ok(lex) = akshar_ime::core::lexicon::Lexicon::from_data(ld) {
                        lex.level(gold, None) > 0
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };

            if !in_lex {
                oov_count += 1;
            }

            let r_key = akshar_ime::core::phonetic_key::collapse_key(roman);
            let g_key = akshar_ime::core::phonetic_key::devanagari_to_collapsed_key(gold);
            let lev = |a: &str, b: &str| -> usize {
                let a: Vec<char> = a.chars().collect();
                let b: Vec<char> = b.chars().collect();
                let mut prev: Vec<usize> = (0..=b.len()).collect();
                let mut curr = vec![0; b.len() + 1];
                for (i, &ca) in a.iter().enumerate() {
                    curr[0] = i + 1;
                    for (j, &cb) in b.iter().enumerate() {
                        let cost = if ca == cb { 0 } else { 1 };
                        curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
                    }
                    prev.copy_from_slice(&curr);
                }
                prev[b.len()]
            };
            if r_key == g_key {
                key_exact_count += 1;
            } else {
                let dist = lev(&r_key, &g_key);
                match dist {
                    1 => key_dist_1 += 1,
                    2 => key_dist_2 += 1,
                    _ => key_dist_3plus += 1,
                }
            }

            if sample_unreachable.len() < 25 {
                sample_unreachable.push((roman.clone(), gold.clone(), in_lex, r_key, g_key));
            }
        }
    }

    println!(
        "\n=== UNREACHABLE DIAGNOSTIC ANALYSIS (Total: {total}, Unreachable: {unreachable}) ==="
    );
    println!(
        "Unreachable rate: {:.2}% ({}/{})",
        unreachable as f64 / total as f64 * 100.0,
        unreachable,
        total
    );
    println!(
        "  - OOV (Gold absent from 700k lexicon): {} ({:.2}%)",
        oov_count,
        oov_count as f64 / unreachable as f64 * 100.0
    );
    println!(
        "  - In lexicon, but key exact: {} ({:.2}%)",
        key_exact_count,
        key_exact_count as f64 / unreachable as f64 * 100.0
    );
    println!(
        "  - In lexicon, key dist=1: {} ({:.2}%)",
        key_dist_1,
        key_dist_1 as f64 / unreachable as f64 * 100.0
    );
    println!(
        "  - In lexicon, key dist=2: {} ({:.2}%)",
        key_dist_2,
        key_dist_2 as f64 / unreachable as f64 * 100.0
    );
    println!(
        "  - In lexicon, key dist>=3: {} ({:.2}%)",
        key_dist_3plus,
        key_dist_3plus as f64 / unreachable as f64 * 100.0
    );
    println!("\nSample unreachable cases:");
    for (r, g, in_lex, rk, gk) in sample_unreachable {
        println!(
            "  roman: {:<20} gold: {:<15} in_lex: {:<5} r_key: {:<15} g_key: {:<15}",
            r, g, in_lex, rk, gk
        );
    }
}

#[test]
#[ignore]
fn diagnose_ranking_errors() {
    let Some(path) = dataset() else { return };
    let Ok(file) = std::fs::File::open(path) else {
        return;
    };
    let mut pairs = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        let Ok(line) = line else { continue };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        let roman = v
            .get("english word")
            .or_else(|| v.get("roman"))
            .and_then(|x| x.as_str());
        let gold = v
            .get("native word")
            .or_else(|| v.get("devanagari"))
            .and_then(|x| x.as_str());
        if let (Some(r), Some(g)) = (roman, gold) {
            pairs.push((r.to_string(), g.to_string()));
        }
    }
    if pairs.is_empty() {
        return;
    };

    let engine = ImeEngine::new();
    let mut ranking_errors = Vec::new();

    for (roman, gold) in &pairs {
        let suggestions = engine.get_suggestions(roman, 8);
        if let Some(pos) = suggestions.iter().position(|(s, _)| s == gold) {
            if pos > 0 {
                let top1 = &suggestions[0];
                let gold_cand = &suggestions[pos];
                ranking_errors.push((
                    roman.clone(),
                    gold.clone(),
                    top1.0.clone(),
                    top1.1,
                    gold_cand.1,
                    pos + 1,
                ));
            }
        }
    }

    let mut gold_key_exact = 0;
    let mut top1_key_exact = 0;
    let mut gold_better_key = 0;

    for (r, g, top1, _, _, _) in &ranking_errors {
        let r_key = akshar_ime::core::phonetic_key::collapse_key(r);
        let g_key = akshar_ime::core::phonetic_key::devanagari_to_collapsed_key(g);
        let t_key = akshar_ime::core::phonetic_key::devanagari_to_collapsed_key(top1);
        if g_key == r_key {
            gold_key_exact += 1;
        }
        if t_key == r_key {
            top1_key_exact += 1;
        }
        if g_key == r_key && t_key != r_key {
            gold_better_key += 1;
        }
    }

    println!(
        "\n=== RANKING ERROR ANALYSIS (Total In-List but Not Top-1: {}) ===",
        ranking_errors.len()
    );
    println!(
        "  Gold has EXACT phonetic key match: {} / {} ({:.2}%)",
        gold_key_exact,
        ranking_errors.len(),
        gold_key_exact as f64 / ranking_errors.len() as f64 * 100.0
    );
    println!(
        "  Top1 has EXACT phonetic key match: {} / {} ({:.2}%)",
        top1_key_exact,
        ranking_errors.len(),
        top1_key_exact as f64 / ranking_errors.len() as f64 * 100.0
    );
    println!(
        "  Gold has EXACT key but Top1 DOES NOT: {} / {} ({:.2}%)",
        gold_better_key,
        ranking_errors.len(),
        gold_better_key as f64 / ranking_errors.len() as f64 * 100.0
    );

    for (r, g, top1, s_top1, s_gold, rank) in ranking_errors.iter().take(20) {
        println!("  roman: {:<18} gold: {:<14} rank: {}/8  top1: {:<14} (top1_score: {}, gold_score: {})",
            r, g, rank, top1, s_top1, s_gold);
    }
}
