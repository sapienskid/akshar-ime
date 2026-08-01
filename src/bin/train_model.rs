// File: src/bin/train_model.rs
//
// M1 training binary: build the generative transliteration model from the
// Aksharantar Nepali corpus and serialise it to a compact binary.
//
// Usage:
//   cargo run --release --bin train_model -- [options]
//     --train <path>    train split (default: data/aksharantar/nep_train.json)
//     --extra <path>    extra split merged in (e.g. data/aksharantar/nep_valid.json)
//     --out <path>      output model path (default: data/translit_model.bin)
//     --limit <n>       only ingest the first <n> clean pairs (debug)
//     --iterations <n>  EM passes (default: 12)
//     --no-seed         skip the deterministic aligner seed
//     --sample <n>      inspect emissions for the top-<n> aksharas

use akshar_ime::core::em_trainer::{Trainer, TrainerConfig};
use akshar_ime::core::translit_model::TranslitModel;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Deserialize)]
struct Record<'a> {
    #[serde(rename = "english word")]
    english: &'a str,
    #[serde(rename = "native word")]
    native: &'a str,
}

struct Args {
    train: PathBuf,
    extra: Option<PathBuf>,
    out: PathBuf,
    limit: Option<usize>,
    iterations: usize,
    seed: bool,
    sample: usize,
}

fn main() {
    let args = parse_args();
    let config = TrainerConfig {
        iterations: args.iterations,
        seed_from_aligner: args.seed,
        limit: args.limit,
        ..Default::default()
    };

    let mut trainer = Trainer::new().with_limit(args.limit);
    let mut skipped = 0usize;
    let start = Instant::now();

    ingest(&args.train, &mut trainer, &mut skipped);
    if let Some(extra) = &args.extra {
        ingest(extra, &mut trainer, &mut skipped);
    }

    eprintln!(
        "Ingested {} clean pairs ({} skipped). Finalising...",
        trainer.ingested, skipped
    );
    let model = trainer.finalize(&config);
    eprintln!("Finalised in {:.1}s", start.elapsed().as_secs_f64());

    let summary = model_summary(&model, args.sample);
    print!("{summary}");

    if let Some(parent) = args.out.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    let t = Instant::now();
    model.save(&args.out).expect("failed to save model");
    let size = std::fs::metadata(&args.out).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "Saved model to {} ({:.1} MB) in {:.1}s",
        args.out.display(),
        size as f64 / 1e6,
        t.elapsed().as_secs_f64()
    );
}

fn ingest(path: &PathBuf, trainer: &mut Trainer, skipped: &mut usize) {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("WARNING: could not open {}: {e}", path.display());
            return;
        }
    };
    let reader = BufReader::new(file);
    let mut count = 0usize;
    let start = Instant::now();

    for line in reader.lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let rec: Record = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(_) => {
                *skipped += 1;
                continue;
            }
        };
        trainer.add_pair(rec.english, rec.native);
        count += 1;
        if count % 500_000 == 0 {
            eprintln!(
                "  [{}] {count} lines ({:.1}s)",
                path.file_name().and_then(|f| f.to_str()).unwrap_or("?"),
                start.elapsed().as_secs_f64()
            );
        }
    }
    eprintln!(
        "  [{}] done: {count} lines in {:.1}s",
        path.file_name().and_then(|f| f.to_str()).unwrap_or("?"),
        start.elapsed().as_secs_f64()
    );
}

fn model_summary(model: &TranslitModel, top_n: usize) -> String {
    let mut emission_entries = 0usize;
    let mut max_emissions = 0usize;
    for list in &model.emissions {
        emission_entries += list.len();
        max_emissions = max_emissions.max(list.len());
    }
    let mut out = String::new();
    out.push_str(&format!(
        "TranslitModel: aksharas={} chunks={} emission_entries={} (max per akshara={}) bigram_entries={}\n",
        model.aksharas.len(),
        model.chunks.len(),
        emission_entries,
        max_emissions,
        model.bigrams.iter().map(|b| b.len()).sum::<usize>()
    ));

    if top_n > 0 {
        let mut by_size: Vec<(usize, usize)> =
            model.emissions.iter().map(|l| l.len()).enumerate().collect();
        by_size.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        out.push_str(&format!("Top {top_n} aksharas by emission count:\n"));
        for (i, n) in by_size.iter().take(top_n) {
            let top = model
                .top_emissions(*i as u32, 5)
                .into_iter()
                .map(|(c, w)| format!("{c}:{:.3}", w))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!("    {} ({n}) [{}]\n", model.aksharas[*i], top));
        }
    }
    out
}

fn parse_args() -> Args {
    let mut train = PathBuf::from("data/aksharantar/nep_train.json");
    let mut extra: Option<PathBuf> = None;
    let mut out = PathBuf::from("data/translit_model.bin");
    let mut limit: Option<usize> = None;
    let mut iterations = 12usize;
    let mut seed = true;
    let mut sample = 15usize;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--train" => train = PathBuf::from(next_value(&arg, args.next())),
            "--extra" => extra = Some(PathBuf::from(next_value(&arg, args.next()))),
            "--out" => out = PathBuf::from(next_value(&arg, args.next())),
            "--limit" => limit = Some(next_value(&arg, args.next()).parse().expect("--limit <n>")),
            "--iterations" => {
                iterations = next_value(&arg, args.next()).parse().expect("--iterations <n>")
            }
            "--no-seed" => seed = false,
            "--sample" => sample = next_value(&arg, args.next()).parse().expect("--sample <n>"),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                eprintln!("Unknown argument: {other}");
                print_help();
                std::process::exit(2);
            }
        }
    }
    Args { train, extra, out, limit, iterations, seed, sample }
}

fn next_value(flag: &str, val: Option<String>) -> String {
    val.unwrap_or_else(|| {
        eprintln!("Missing value for {flag}");
        std::process::exit(2);
    })
}

fn print_help() {
    println!("Usage: cargo run --release --bin train_model -- [options]");
    println!("  --train <path>    train split (default: data/aksharantar/nep_train.json)");
    println!("  --extra <path>    extra split merged in");
    println!("  --out <path>      output model path (default: data/translit_model.bin)");
    println!("  --limit <n>       only ingest first <n> clean pairs (debug)");
    println!("  --iterations <n>  EM passes (default: 12)");
    println!("  --no-seed         skip the deterministic aligner seed");
    println!("  --sample <n>      inspect top-<n> aksharas by emission count");
    println!("  -h, --help        show help");
}
