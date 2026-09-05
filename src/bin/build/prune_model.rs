// File: src/bin/prune_model.rs
//
// Phonetic-model pruning: after training emissions on all Devanagari data
// (Hindi included), drop the emission rows of aksharas that never appear in
// the Nepali vocabulary — they cannot be correct outputs for a Nepali IME,
// and their peaked emissions crowd real candidates out of the reverse index.
//
// Usage: prune_model --model data/translit_model.bin \
//                    --vocab data/word_freq_text.bin \
//                    --min-count 1 [--out data/translit_model.bin]

use akshar_ime::core::akshara::segment;
use akshar_ime::core::translit_model::TranslitModel;
use std::collections::HashSet;
use std::path::Path;

fn main() {
    let mut model_path = "data/translit_model.bin".to_string();
    let mut vocab_path = "data/word_freq_text.bin".to_string();
    let mut out_path = "data/translit_model.bin".to_string();
    let mut min_count = 1u32;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model_path = args.next().expect("v"),
            "--vocab" => vocab_path = args.next().expect("v"),
            "--out" => out_path = args.next().expect("v"),
            "--min-count" => min_count = args.next().expect("v").parse().unwrap(),
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    let mut model = TranslitModel::load(Path::new(&model_path)).expect("load model");
    // Aksharas that occur in the Nepali vocabulary (the IME's target words).
    let map: std::collections::HashMap<String, u32> =
        bincode::deserialize(&std::fs::read(&vocab_path).expect("vocab")).expect("deserialize");
    let mut seen: HashSet<u32> = HashSet::new();
    for (word, &count) in &map {
        if count < min_count {
            continue;
        }
        for a in segment(word) {
            if let Some(id) = model.akshara_id(&a) {
                seen.insert(id);
            }
        }
    }
    let before: usize = model.emissions.iter().filter(|e| !e.is_empty()).count();
    for (a, em) in model.emissions.iter_mut().enumerate() {
        if !seen.contains(&(a as u32)) {
            em.clear();
        }
    }
    let after: usize = model.emissions.iter().filter(|e| !e.is_empty()).count();
    // chunks still referenced stay; unreferenced ones are harmless.
    model.save(Path::new(&out_path)).expect("save");
    eprintln!(
        "pruned emissions: {} -> {} aksharas with candidates (vocabulary words: {})",
        before,
        after,
        map.len()
    );
}
