// File: src/bin/evaluate_model_v2.rs
//
// M3 gate: decode-only benchmark of the v2 pair-grammar decoder on the
// Aksharantar Nepali test split, same buckets as evaluate_model.
//
// Usage: cargo run --release --bin evaluate_model_v2 -- [--model p] [--dataset p]

use akshar_ime::core::v2::{PairDecoder, PairModel};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Deserialize)]
struct Record<'a> {
    #[serde(rename = "english word")]
    roman: &'a str,
    #[serde(rename = "native word")]
    target: &'a str,
    source: &'a str,
}

fn main() {
    let mut model_path = PathBuf::from("data/pair_model_v2.bin");
    let mut dataset = PathBuf::from("data/aksharantar/nep_test.json");
    let mut probe: Option<String> = None;
    let mut trans: Option<String> = None;
    let mut pair_weight: f64 = 0.5;
    // M4 pilot: vocabulary rescoring (freq bonus for real corpus words).
    let mut vocab_weight: f64 = 0.0;
    let mut vocab_min_count: u32 = 1;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--probe" => {
                probe = args.next();
            }
            "--trans" => trans = Some(args.next().expect("chunk")),
            "--pair-weight" => pair_weight = args.next().expect("f").parse().expect("f"),
            "--vocab-weight" => vocab_weight = args.next().expect("f").parse().expect("f"),
            "--vocab-min" => vocab_min_count = args.next().expect("n").parse().expect("n"),
            "--model" => model_path = args.next().expect("path").into(),
            "--dataset" => dataset = args.next().expect("path").into(),
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    let model = PairModel::load(&model_path).expect("load v2 model");
    let mut decoder = PairDecoder::new(model);
    decoder.pair_weight = pair_weight;

    // Load the vocabulary map (bincode HashMap<String,u32> from build_wordfreq).
    let vocab: Option<std::collections::HashMap<String, u32>> = if vocab_weight > 0.0 {
        match std::fs::read("data/word_freq.bin") {
            Ok(bytes) => match bincode::deserialize::<std::collections::HashMap<String, u32>>(&bytes) {
                Ok(map) => {
                    let pruned: std::collections::HashMap<String, u32> = map
                        .into_iter()
                        .filter(|(_, c)| *c >= vocab_min_count)
                        .collect();
                    eprintln!(
                        "vocab: {} words (min count {vocab_min_count})",
                        pruned.len()
                    );
                    Some(pruned)
                }
                Err(e) => {
                    eprintln!("WARNING: word_freq.bin deserialize failed: {e}");
                    None
                }
            },
            Err(e) => {
                eprintln!("WARNING: word_freq.bin not found: {e}");
                None
            }
        }
    } else {
        None
    };

    // Debug: --probe <word> prints the k-best lattice paths and exits.
    let probe = std::env::args().position(|a| a == "--probe").and_then(|i| std::env::args().nth(i + 1));
    if let Some(word) = probe {
        if let Some(chunks) = word.strip_prefix("rev:") {
            for cs in chunks.split(',') {
                println!("chunk {cs:?}: {:?}", decoder.model.chunk_candidates_raw(cs.as_bytes()));
            }
            return;
        }
        // Side-by-side with the v1 decoder on the same word.
        if let Ok(v1model) =
            akshar_ime::core::translit_model::TranslitModel::load(std::path::Path::new(
                "data/translit_model.bin",
            ))
        {
            let v1 = akshar_ime::core::decoder::ModelDecoder::new(v1model);
            println!("v1 decode of {word:?}:");
            for (dev, _s) in v1.decode(&word, 5) {
                println!("  {dev}");
            }
        }
        println!("v2 decode of {word:?}:");
        for c in decoder.decode(&word, 10) {
            println!(
                "{:<20} emit={:.3} lm={:.3} score={:.3}",
                c.dev,
                c.emit,
                c.lm,
                c.emit + c.lm
            );
        }
        return;
    }

    if let Some(cs) = trans {
        use akshar_ime::core::translit_model::pack_chunk_bytes;
        use akshar_ime::core::v2::{pair_akshara, pair_chunk};
        let ckey = pack_chunk_bytes(cs.as_bytes());
        let curs: Vec<u64> = decoder
            .model
            .emit_w
            .keys()
            .filter(|&&k| pair_chunk(k) == ckey)
            .copied()
            .collect();
        let mut rows: Vec<(f32, String)> = Vec::new();
        let from_na: Vec<u64> = decoder
            .model
            .emit_w
            .keys()
            .filter(|&&k| pair_chunk(k) == pack_chunk_bytes(b"ma"))
            .copied()
            .collect();
        println!(
            "pairs emitting chunk \"ma\": {}; of those, appearing as PREV in bi: {}",
            from_na.len(),
            from_na.iter().filter(|k| decoder.model.bi.keys().any(|(p, _)| p == *k)).count()
        );
        let mut rows: Vec<(f32, String)> = Vec::new();
        for (&(prev, cur), &w) in &decoder.model.bi {
            if curs.contains(&cur) && from_na.contains(&prev) && pair_chunk(prev) == pack_chunk_bytes(b"ma") {
                let a = decoder
                    .model
                    .aksharas
                    .get(pair_akshara(prev) as usize)
                    .cloned()
                    .unwrap_or_default();
                rows.push((
                    w,
                    format!(
                        "{}+{:?} -> w={:.2}",
                        a,
                        decoder.model.chunk_string(pair_chunk(prev)),
                        w
                    ),
                ));
            }
        }
        rows.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (_, s) in rows.iter().take(10) {
            println!("{s}");
        }
        println!(
            "(bi has {} entries total; {} cur pairs emit {:?})",
            decoder.model.bi.len(),
            curs.len(),
            cs
        );
        return;
    }
    println!("IndicXlit (neural, top-1) reference: native=80.25%, named-entities=52.67%");

    let f = std::fs::File::open(&dataset).expect("open dataset");
    let mut stats: HashMap<&str, [usize; 4]> = HashMap::new(); // total, top1, top5, latency_us
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<Record>(trimmed) else {
            continue;
        };
        let roman = rec.roman.to_ascii_lowercase();
        let target = rec.target.trim();
        if roman.is_empty() || target.is_empty() {
            continue;
        }
        let t0 = Instant::now();
        let mut cands = decoder.decode(&roman, 5);
        if let Some(vocab) = &vocab {
            for c in &mut cands {
                if let Some(&f) = vocab.get(&c.dev) {
                    c.lm -= vocab_weight * (1.0 + f as f64).ln();
                }
            }
            cands.sort_by(|a, b| {
                (a.emit + a.lm).partial_cmp(&(b.emit + b.lm)).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        let us = t0.elapsed().as_micros() as usize;
        let bucket = match rec.source {
            "AK-Freq" => "AK-Freq",
            "AK-NEF" | "AK-NEI" => "NE",
            _ => "other",
        };
        let e = stats.entry("ALL").or_insert([0, 0, 0, 0]);
        e[0] += 1;
        e[3] += us;
        if cands.first().map_or(false, |c| c.dev == target) {
            e[1] += 1;
        }
        if cands.iter().any(|c| c.dev == target) {
            e[2] += 1;
        }
        if bucket != "other" {
            let e = stats.entry(bucket).or_insert([0, 0, 0, 0]);
            e[0] += 1;
            e[3] += us;
            if cands.first().map_or(false, |c| c.dev == target) {
                e[1] += 1;
            }
            if cands.iter().any(|c| c.dev == target) {
                e[2] += 1;
            }
        }
    }

    for name in ["ALL", "AK-Freq", "NE"] {
        if let Some([total, t1, t5, us]) = stats.get(name) {
            let t = *total as f64;
            println!(
                "  {name:<10} top1={:.2}%  top5={:.2}%  ({t1}/{total})  {:.2} ms/word",
                *t1 as f64 / t * 100.0,
                *t5 as f64 / t * 100.0,
                *us as f64 / t / 1000.0
            );
        }
    }
}
