// File: src/bin/probe_model.rs
//
// Diagnostic: dump the decoder's behaviour on a single roman word.
//
// Usage: cargo run --release --bin probe_model -- <word> [--model <path>]

use akshar_ime::core::decoder::ModelDecoder;
use akshar_ime::core::translit_model::TranslitModel;
use std::path::Path;

fn main() {
    let mut word = "holi".to_string();
    let mut model_path = "data/translit_model.bin".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_path = args.next().unwrap_or_else(|| model_path.clone()),
            "-h" | "--help" => {
                println!("usage: probe_model <word> [--model path]");
                return;
            }
            other => word = other.to_string(),
        }
    }

    let model = TranslitModel::load(Path::new(&model_path)).expect("load model");
    let dec = ModelDecoder::new(model);

    let roman = word.to_ascii_lowercase();
    let bytes = roman.as_bytes();

    // Dump reverse-index candidates per position.
    for i in 0..roman.len() {
        for l in 1..=4.min(roman.len() - i) {
            let chunk = &roman[i..i + l];
            let list = dec.chunk_candidates(chunk);
            let shown: Vec<String> = list
                .iter()
                .take(8)
                .map(|(a, w)| format!("{}({:.2})", dec.model.aksharas[*a as usize], w))
                .collect();
            println!("pos {i} chunk `{chunk}` -> {}", shown.join(" "));        }
    }

    println!("\nTop 10 decodings:");
    for (d, w) in dec.decode(&roman, 10) {
        println!("  {d}  (weight {w:.3})");
    }

    println!("\nKN stats:");
    for aks in ["हो", "ली", "ओः", "ऊः", "ग्को", "को", "ल"] {
        let Some(aid) = dec.model.akshara_id(aks) else {
            println!("  {aks}: not in vocab");
            continue;
        };
        let uni = dec.model.unigram_kn[aid as usize];
        let backoff = dec.model.backoff[aid as usize];
        let n_bi = dec.model.bigrams[aid as usize].len();
        println!(
            "  {aks}: id={aid} unigram_kn={uni:.3} backoff={backoff:.3} bigrams={n_bi}"
        );
    }
    let _ = bytes;
}
