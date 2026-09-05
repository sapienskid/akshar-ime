// File: src/bin/romanize.rs
//
// E5 v2: backward romanization driven by the trained model itself.
//
// For each vocabulary word: segment into aksharas (engine segmenter), then
// per akshara take the top-K roman chunks from the EM emission table
// P(chunk | akshara).  The argmax chunk spells the canonical form; runner-up
// chunks are the *realistic* spelling variants (they are variants actual
// users produced in the corpus).  Combinations are capped and weighted by
// joint probability, so noisy spellings are learned without noisy data.
//
// Usage:
//   cargo run --release --bin romanize -- \
//     [--vocab data-pipeline/out/word_freq.csv] \
//     [--model data/translit_model.bin] \
//     [--out data-pipeline/out/synthetic.jsonl] \
//     [--max-combos 8] [--top 3] [--max-weight 9.0]

use akshar_ime::core::akshara::segment;
use akshar_ime::core::translit_model::TranslitModel;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

fn main() {
    let mut vocab_path = "data-pipeline/out/word_freq.csv".to_string();
    let mut model_path = "data/translit_model.bin".to_string();
    let mut out_path = "data-pipeline/out/synthetic.jsonl".to_string();
    let mut max_combos = 8usize;
    let mut top = 3usize;
    let mut max_weight = 9.0f32;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--vocab" => vocab_path = args.next().expect("v"),
            "--model" => model_path = args.next().expect("v"),
            "--out" => out_path = args.next().expect("v"),
            "--max-combos" => max_combos = args.next().expect("v").parse().unwrap(),
            "--top" => top = args.next().expect("v").parse().unwrap(),
            "--max-weight" => max_weight = args.next().expect("v").parse().unwrap(),
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    let model = TranslitModel::load(Path::new(&model_path)).expect("load model");
    // akshara string -> candidate roman chunks sorted by -log P ascending.
    let mut cands: HashMap<u32, Vec<(String, f32)>> = HashMap::new();
    for (a, list) in model.emissions.iter().enumerate() {
        let mut v: Vec<(String, f32)> = list
            .iter()
            .filter_map(|(cid, w)| {
                model.chunks.get(*cid as usize).map(|s| (s.clone(), *w))
            })
            .filter(|(_, w)| *w <= max_weight && *w > 0.0)
            .collect();
        v.sort_by(|x, y| x.1.total_cmp(&y.1));
        v.truncate(top);
        if !v.is_empty() {
            cands.insert(a as u32, v);
        }
    }
    eprintln!("candidates for {} aksharas", cands.len());

    let f = std::fs::File::open(&vocab_path).expect("open vocab csv");
    let mut out = std::io::BufWriter::new(std::fs::File::create(&out_path).expect("create out"));
    let mut n_pairs = 0usize;
    let mut n_words = 0usize;
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        let line = line.trim();
        if line.is_empty() || line.starts_with("word,") {
            continue;
        }
        let Some((word, freq)) = line.rsplit_once(',') else {
            continue;
        };
        let Ok(freq) = freq.parse::<u64>() else { continue };
        let aks = segment(word);
        if aks.is_empty() {
            continue;
        }
        // per-akshara candidate lists; skip words with unknown aksharas
        let mut lists: Vec<&Vec<(String, f32)>> = Vec::with_capacity(aks.len());
        let mut ok = true;
        for a in &aks {
            match model.akshara_id(a).and_then(|id| cands.get(&id)) {
                Some(c) => lists.push(c),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            continue;
        }
        // combine: keep the max_combos lowest-weight joint spellings
        let mut combos: Vec<(String, f32)> = vec![(String::new(), 0.0)];
        for list in &lists {
            let mut next: Vec<(String, f32)> = Vec::new();
            for (s, w) in &combos {
                for (c, cw) in list.iter().take(top) {
                    next.push((format!("{s}{c}"), w + cw));
                }
            }
            next.sort_by(|a, b| a.1.total_cmp(&b.1));
            next.truncate(max_combos);
            combos = next;
        }
        n_words += 1;
        for (rank, (roman, w)) in combos.iter().enumerate() {
            // weight: canonical gets the word frequency, variants decay
            let reps = if rank == 0 {
                freq.min(4)
            } else {
                (freq as f64 * (-*w as f64).exp() * 4.0).round() as u64
            }
            .max(1)
            .min(2);
            for _ in 0..reps {
                writeln!(
                    out,
                    "{}",
                    json_line(&roman.to_lowercase(), word)
                )
                .expect("write");
                n_pairs += 1;
            }
        }
    }
    out.flush().expect("flush");
    eprintln!("synthetic pairs: {n_pairs} from {n_words} words -> {out_path}");
}

fn json_line(roman: &str, native: &str) -> String {
    format!(
        "{{\"english\": {}, \"native\": {}}}",
        serde_json::to_string(roman).unwrap(),
        serde_json::to_string(native).unwrap()
    )
}
