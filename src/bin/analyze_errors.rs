// File: src/bin/analyze_errors.rs
//
// E0 error-analysis harness: where does the engine lose accuracy?
//
// Reports, per source bucket (AK-Freq = native words, AK-NEF/AK-NEI = named
// entities) over the Aksharantar Nepali test split:
//   1. Oracle top-k curves for the decoder and the full engine
//      (gold-in-first-k for k = 1,2,3,5,8,10,20,50) -> splits RANKING error
//      (gold generated but not ranked first) from COVERAGE error (gold never
//      generated).
//   2. Engine-vs-decoder top-1 disagreement: when they differ, who is right?
//      Quantifies fusion-layer damage.
//   3. Strict vs multi-reference top-1 (any native observed for the same roman
//      in the training split counts as correct) -> ambiguity ceiling.
//   4. Error taxonomy of decoder top-1 misses: matra-only (vowel length /
//      nasal), halant-only (conjunct), or substantive (coverage).
//   5. Character error rate (CER) of the top-1 candidate.
//
// Usage:
//   cargo run --release --bin analyze_errors -- [options]
//     --test <path>   test split (default: data/aksharantar/nep_test.json)
//     --train <path>  train split for multi-reference sets (default:
//                     data/aksharantar/nep_train.json)

use akshar_ime::ImeEngine;
use akshar_ime::core::decoder::{DecoderConfig, ModelDecoder};
use akshar_ime::core::translit_model::TranslitModel;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Deserialize)]
struct Record<'a> {
    #[serde(rename = "english word")]
    roman: &'a str,
    #[serde(rename = "native word")]
    target: &'a str,
    source: &'a str,
}

struct Case {
    roman: String,
    target: String,
    source: String,
    refs: HashSet<String>,
}

const KS: [usize; 8] = [1, 2, 3, 5, 8, 10, 20, 50];

fn main() {
    let mut test_path = "data/aksharantar/nep_test.json".to_string();
    let mut train_path = "data/aksharantar/nep_train.json".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--test" => test_path = args.next().expect("value"),
            "--train" => train_path = args.next().expect("value"),
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    let cases = load_test(&test_path);
    let n = cases.len();
    println!("cases: {n}");

    // Multi-reference sets from the training split (one streaming pass).
    let refsets = build_refsets(&train_path, &cases);
    let mut cases = cases;
    for c in &mut cases {
        c.refs.insert(c.target.clone());
        if let Some(set) = refsets.get(&c.roman) {
            for r in set {
                c.refs.insert(r.clone());
            }
        }
    }
    let ambiguous = cases.iter().filter(|c| c.refs.len() > 1).count();
    println!(
        "multi-reference romans: {ambiguous}/{n} ({:.1}%)",
        ambiguous as f64 / n as f64 * 100.0
    );

    let model = TranslitModel::load(Path::new("data/translit_model.bin")).expect("load model");
    let decoder = ModelDecoder::with_config(
        model,
        DecoderConfig {
            beam_width: 64,
            ..Default::default()
        },
    );
    let engine = ImeEngine::new();

    // Buckets: AK-Freq (native), NE (NEF+NEI), ALL.
    let bucket = |src: &str| match src {
        "AK-Freq" => "AK-Freq",
        "AK-NEF" | "AK-NEI" => "NE",
        _ => "other",
    };

    struct Agg {
        name: &'static str,
        total: usize,
        dec_hits: [usize; KS.len()],      // strict decoder oracle
        dec_hits_mr: [usize; KS.len()],   // multi-ref decoder oracle
        eng_hits: [usize; KS.len()],      // strict engine oracle
        dec_top1: usize,
        dec_top1_mr: usize,
        eng_top1: usize,
        eng_top1_mr: usize,
        dis_dec_right: usize,
        dis_eng_right: usize,
        dis_both_wrong: usize,
        tax_matra: usize,
        tax_halant: usize,
        tax_subst: usize,
        cer_num: usize,
        cer_den: usize,
    }
    let mk = |name: &'static str| Agg {
        name,
        total: 0,
        dec_hits: [0; KS.len()],
        dec_hits_mr: [0; KS.len()],
        eng_hits: [0; KS.len()],
        dec_top1: 0,
        dec_top1_mr: 0,
        eng_top1: 0,
        eng_top1_mr: 0,
        dis_dec_right: 0,
        dis_eng_right: 0,
        dis_both_wrong: 0,
        tax_matra: 0,
        tax_halant: 0,
        tax_subst: 0,
        cer_num: 0,
        cer_den: 0,
    };
    let mut aggs = [mk("ALL"), mk("AK-Freq"), mk("NE")];

    for case in &cases {
        let b_idx = match bucket(&case.source) {
            "AK-Freq" => 1,
            _ => 2, // fold NE + rare buckets together
        };
        let dec: Vec<String> = decoder
            .decode_detailed(&case.roman, 50)
            .into_iter()
            .map(|c| c.dev)
            .collect();
        let eng: Vec<String> = engine
            .get_suggestions(&case.roman, 10)
            .into_iter()
            .map(|(d, _)| d)
            .collect();

        for idx in [0usize, b_idx] {
            let agg = &mut aggs[idx];
            agg.total += 1;
            for (i, k) in KS.iter().enumerate() {
                if dec.get(..*k).map_or(false, |p| p.contains(&case.target)) {
                    agg.dec_hits[i] += 1;
                }
                if dec.get(..*k).map_or(false, |p| p.iter().any(|c| case.refs.contains(c))) {
                    agg.dec_hits_mr[i] += 1;
                }
                if eng.get(..*k).map_or(false, |p| p.contains(&case.target)) {
                    agg.eng_hits[i] += 1;
                }
            }
            if dec.first().map_or(false, |c| *c == case.target) {
                agg.dec_top1 += 1;
            }
            if dec.first().map_or(false, |c| case.refs.contains(c)) {
                agg.dec_top1_mr += 1;
            }
            if eng.first().map_or(false, |c| *c == case.target) {
                agg.eng_top1 += 1;
            }
            if eng.first().map_or(false, |c| case.refs.contains(c)) {
                agg.eng_top1_mr += 1;
            }
            // Disagreement analysis.
            let dt = dec.first();
            let et = eng.first();
            if dt != et {
                let d_right = dt.map_or(false, |c| case.refs.contains(c));
                let e_right = et.map_or(false, |c| case.refs.contains(c));
                if d_right && !e_right {
                    agg.dis_dec_right += 1;
                } else if e_right && !d_right {
                    agg.dis_eng_right += 1;
                } else {
                    agg.dis_both_wrong += 1;
                }
            }
            // Taxonomy + CER on decoder top-1 misses (strict).
            let miss = dec.first().map_or(true, |c| *c != case.target);
            if miss {
                if let Some(pred) = dec.first() {
                    match classify(pred, &case.target) {
                        Tax::Matra => agg.tax_matra += 1,
                        Tax::Halant => agg.tax_halant += 1,
                        Tax::Substantive => agg.tax_subst += 1,
                    }
                    agg.cer_num += levenshtein(pred, &case.target);
                }
            }
            agg.cer_den += case.target.chars().count();
        }
    }

    for agg in &aggs {
        let t = agg.total.max(1) as f64;
        println!("\n=== {} (n={}) ===", agg.name, agg.total);
        println!("decoder top-1: strict {:.2}%  multi-ref {:.2}%", agg.dec_top1 as f64 / t * 100.0, agg.dec_top1_mr as f64 / t * 100.0);
        println!("engine  top-1: strict {:.2}%  multi-ref {:.2}%", agg.eng_top1 as f64 / t * 100.0, agg.eng_top1_mr as f64 / t * 100.0);
        print!("decoder oracle k: ");
        for (i, k) in KS.iter().enumerate() {
            print!("{}:{:.1}% ", k, agg.dec_hits[i] as f64 / t * 100.0);
        }
        println!();
        print!("decoder oracle k (multi-ref): ");
        for (i, k) in KS.iter().enumerate() {
            print!("{}:{:.1}% ", k, agg.dec_hits_mr[i] as f64 / t * 100.0);
        }
        println!();
        print!("engine oracle k: ");
        for (i, k) in KS.iter().enumerate() {
            print!("{}:{:.1}% ", k, agg.eng_hits[i] as f64 / t * 100.0);
        }
        println!();
        println!(
            "disagreements (dec_right/eng_right/both_wrong): {}/{}/{}",
            agg.dis_dec_right, agg.dis_eng_right, agg.dis_both_wrong
        );
        println!(
            "decoder-miss taxonomy: matra-only {}  halant-only {}  substantive {}",
            agg.tax_matra, agg.tax_halant, agg.tax_subst
        );
        println!("CER (decoder top-1): {:.2}%", agg.cer_num as f64 / agg.cer_den.max(1) as f64 * 100.0);
    }
}

fn load_test(path: &str) -> Vec<Case> {
    let f = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut out = Vec::new();
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<Record>(trimmed) else {
            continue;
        };
        let roman = rec.roman.to_ascii_lowercase();
        let target = rec.target.trim().to_string();
        if roman.is_empty() || target.is_empty() {
            continue;
        }
        out.push(Case {
            roman,
            target,
            source: rec.source.to_string(),
            refs: HashSet::new(),
        });
    }
    out
}

/// Streaming pass over the train split: collect native variants only for the
/// romans that appear in the test set (keeps memory tiny).
fn build_refsets(path: &str, cases: &[Case]) -> HashMap<String, HashSet<String>> {
    let want: HashSet<String> = cases.iter().map(|c| c.roman.clone()).collect();
    let mut map: HashMap<String, HashSet<String>> = HashMap::new();
    let f = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("WARNING: cannot open train split {path}: {e}; multi-ref analysis skipped");
            return map;
        }
    };
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<Record>(trimmed) else {
            continue;
        };
        let roman = rec.roman.to_ascii_lowercase();
        if want.contains(&roman) {
            map.entry(roman)
                .or_default()
                .insert(rec.target.trim().to_string());
        }
    }
    map
}

enum Tax {
    Matra,
    Halant,
    Substantive,
}

fn is_matra(c: char) -> bool {
    matches!(c, '\u{0901}'..='\u{0903}' | '\u{093C}' | '\u{093E}'..='\u{094C}')
}

fn is_halant(c: char) -> bool {
    c == '\u{094D}'
}

fn strip_matching(s: &str, matra: bool, halant: bool) -> String {
    s.chars()
        .filter(|&c| {
            let m = is_matra(c);
            let h = is_halant(c);
            !((matra && m) || (halant && h))
        })
        .collect()
}

/// Compare a wrong prediction against the gold form.
fn classify(pred: &str, gold: &str) -> Tax {
    if strip_matching(pred, true, true) == strip_matching(gold, true, true) {
        return Tax::Matra;
    }
    if strip_matching(pred, false, true) == strip_matching(gold, false, true) {
        return Tax::Halant;
    }
    Tax::Substantive
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}
