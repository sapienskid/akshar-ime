// File: src/bin/train_pair_model.rs
//
// Train the pair-bigram grammar (context-dependent transitions over
// aligned akshara/chunk pairs) and save it as data/pair_model.bin.
//
// The EM loop is identical to train_model (the unigram emissions become the
// pair backoff distribution); the extra step collects transition posteriors.

use akshar_ime::core::em_trainer::{Trainer, TrainerConfig};
use serde::Deserialize;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

#[derive(Deserialize)]
struct Record<'a> {
    #[serde(rename = "english word", alias = "english")]
    english: &'a str,
    #[serde(rename = "native word", alias = "native")]
    native: &'a str,
}

fn main() {
    let mut paths: Vec<PathBuf> = vec!["data/aksharantar/nep_train.json".into()];
    let mut out = PathBuf::from("data/pair_model.bin");
    let mut alpha_pair = 0.5f64;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--extra" => paths.push(args.next().expect("path").into()),
            "--out" => out = args.next().expect("path").into(),
            "--alpha" => alpha_pair = args.next().expect("f").parse().expect("f"),
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    let config = TrainerConfig::default();
    let mut trainer = Trainer::new();
    let mut skipped = 0usize;
    for path in &paths {
        ingest(path, &mut trainer, &mut skipped);
    }
    eprintln!("Ingested {} clean pairs ({skipped} skipped)", trainer.ingested);

    let model = trainer.finalize_pair(&config, alpha_pair);
    let bytes = bincode::serialize(&model).expect("serialize");
    std::fs::write(&out, &bytes).expect("write");
    eprintln!(
        "Saved {} ({:.1} MB): {} aksharas, {} uni, {} bi, {} bi_ak",
        out.display(),
        bytes.len() as f64 / 1e6,
        model.aksharas.len(),
        model.emit_w.len(),
        model.bi.len(),
        model.bi_ak.len()
    );
}

fn ingest(path: &PathBuf, trainer: &mut Trainer, skipped: &mut usize) {
    let Ok(file) = std::fs::File::open(path) else {
        eprintln!("WARNING: cannot open {}", path.display());
        return;
    };
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<Record>(trimmed) else {
            *skipped += 1;
            continue;
        };
        trainer.add_pair(rec.english, rec.native);
    }
}
