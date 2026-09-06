// File: src/bin/build/prune_lm.rs
//
// Relative-entropy pruning of the syllable n-gram LM.
//
// The existing pruning knobs are frequency thresholds (`--bigram-min-freq`,
// vocabulary `--min-freq`).  A count cutoff is a poor proxy for usefulness: it
// deletes rare-but-informative transitions and keeps frequent-but-predictable
// ones.  The question that actually matters is how much a transition changes
// the model's predictions, which is what Stolcke pruning measures.
//
// A trigram (a,b -> c) is redundant when its weight is close to what the model
// would say anyway after backing off:
//
//     w_backoff(a,b -> c) = trigram_backoff(a,b) + bigram_weight(b -> c)
//
// Dropping such an entry costs almost nothing, because the backoff path
// reproduces it.  The cost of dropping is weighted by how often the entry is
// actually consulted, approximated by its own probability exp(-w) — an entry
// that is both rare and well-predicted by backoff is free to remove, while a
// common one is not, however close the two weights are:
//
//     contribution = exp(-w) * |w - w_backoff|
//
// This is the Stolcke criterion with the training counts replaced by the
// model's own probabilities, which is what a pack-time tool has available.
//
// Usage:
//   cargo run --release --bin prune_lm -- \
//     --model data/akshar.model --out data/akshar_pruned.model \
//     --trigram-threshold 1e-6 [--bigram-threshold 0] [--dry-run]

use akshar_ime::core::unified::UnifiedModel;
use std::path::Path;

fn main() {
    let mut model_path = "data/akshar.model".to_string();
    let mut out_path = "data/akshar_pruned.model".to_string();
    let mut tri_threshold = 1e-6f64;
    let mut bi_threshold = 0.0f64;
    let mut dry_run = false;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model_path = args.next().expect("value for --model"),
            "--out" => out_path = args.next().expect("value for --out"),
            "--trigram-threshold" => {
                tri_threshold = args.next().expect("value").parse().expect("float")
            }
            "--bigram-threshold" => {
                bi_threshold = args.next().expect("value").parse().expect("float")
            }
            "--dry-run" => dry_run = true,
            "--help" | "-h" => {
                println!("prune_lm — relative-entropy pruning of the syllable n-gram LM");
                println!("  --model <path>              input model");
                println!("  --out <path>                output model");
                println!("  --trigram-threshold <f>     drop trigrams below this contribution");
                println!("  --bigram-threshold <f>      drop bigrams below this contribution");
                println!("  --dry-run                   report only, do not write");
                return;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    let mut m = UnifiedModel::load(Path::new(&model_path)).expect("load model");
    let t = &mut m.translit;

    let tri_before: usize = t.trigrams.iter().map(|r| r.len()).sum();
    let bi_before: usize = t.bigrams.iter().map(|r| r.len()).sum();

    // --- trigrams -----------------------------------------------------------
    // Snapshot the bigram tables first: the backoff estimate must be computed
    // against the unpruned bigram model, otherwise pruning the two together
    // would make each decision depend on the order they happen in.
    let bigrams_snapshot = t.bigrams.clone();
    let backoff_snapshot = t.backoff.clone();
    let unigram_snapshot = t.unigram_kn.clone();

    let bigram_weight = |a: u32, b: u32| -> f64 {
        if let Some(list) = bigrams_snapshot.get(a as usize) {
            if let Some((_, w)) = list.iter().find(|(id, _)| *id == b) {
                return *w as f64;
            }
        }
        backoff_snapshot.get(a as usize).copied().unwrap_or(0.0) as f64
            + unigram_snapshot.get(b as usize).copied().unwrap_or(0.0) as f64
    };

    if tri_threshold > 0.0 {
        for (i, &(_, b)) in t.trigram_keys.iter().enumerate() {
            let backoff = t.trigram_backoff.get(i).copied().unwrap_or(0.0) as f64;
            t.trigrams[i].retain(|&(c, w)| {
                let w = w as f64;
                let w_backoff = backoff + bigram_weight(b, c);
                let contribution = (-w).exp() * (w - w_backoff).abs();
                contribution >= tri_threshold
            });
        }
    }

    // --- bigrams ------------------------------------------------------------
    // Same criterion one order down: the fallback for a bigram is its own
    // backoff plus the continuation unigram.
    if bi_threshold > 0.0 {
        for a in 0..t.bigrams.len() {
            let backoff = t.backoff.get(a).copied().unwrap_or(0.0) as f64;
            let unigram = unigram_snapshot.clone();
            t.bigrams[a].retain(|&(b, w)| {
                let w = w as f64;
                let w_backoff = backoff + unigram.get(b as usize).copied().unwrap_or(0.0) as f64;
                let contribution = (-w).exp() * (w - w_backoff).abs();
                contribution >= bi_threshold
            });
        }
    }

    let tri_after: usize = t.trigrams.iter().map(|r| r.len()).sum();
    let bi_after: usize = t.bigrams.iter().map(|r| r.len()).sum();

    println!(
        "trigram transitions: {tri_before} -> {tri_after}  ({:.1}% kept)",
        pct(tri_after, tri_before)
    );
    println!(
        "bigram  transitions: {bi_before} -> {bi_after}  ({:.1}% kept)",
        pct(bi_after, bi_before)
    );

    // Contexts left with no successors still cost a key and a backoff weight
    // each, and the decoder reaches the same answer through the bigram path,
    // so drop them outright.
    let keep: Vec<bool> = t.trigrams.iter().map(|r| !r.is_empty()).collect();
    let dropped_contexts = keep.iter().filter(|k| !**k).count();
    if dropped_contexts > 0 {
        let mut it = keep.iter();
        t.trigrams.retain(|_| *it.next().unwrap());
        let mut it = keep.iter();
        t.trigram_keys.retain(|_| *it.next().unwrap());
        let mut it = keep.iter();
        t.trigram_backoff.retain(|_| *it.next().unwrap());
        println!("empty trigram contexts dropped: {dropped_contexts}");
    }
    t.build_trigram_index();

    if dry_run {
        println!("(dry run — not written)");
        return;
    }
    m.save(Path::new(&out_path)).expect("save model");
    let size = std::fs::metadata(&out_path).expect("stat").len();
    println!("Wrote {out_path} ({:.2} MB)", size as f64 / 1048576.0);
}

fn pct(a: usize, b: usize) -> f64 {
    if b == 0 {
        0.0
    } else {
        a as f64 / b as f64 * 100.0
    }
}
