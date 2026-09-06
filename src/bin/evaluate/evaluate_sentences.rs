// File: src/bin/evaluate/evaluate_sentences.rs
//
// Sentence-level (in-context) evaluation for Akshar IME.
//
// The Aksharantar benchmark scores 4,101 *isolated* words, so it never
// supplies a preceding word and the corpus word-bigram table — 29% of the
// model file — contributes exactly 0.00% to it by construction.  Real typing
// is nothing like that.  This harness measures the thing the bigram table is
// actually for.
//
// Method:
//   1. Take held-out sentences from data/store/corpus_clean.txt.  The split is
//      the stable hash in core::holdout, the same one the trainer excludes, so
//      these sentences are never in vocab counts, bigrams or the EM pairs.
//   2. Romanize each Devanagari word by segmenting it into aksharas and
//      emitting each akshara's single most likely roman chunk under the
//      trained EM emissions.  That is the spelling the model itself considers
//      most natural, which makes this a measure of ranking rather than of
//      romanization guesswork.
//   3. Decode the sentence left to right, feeding each *gold* word back as
//      context (teacher forcing, via set_context_word, which does not learn).
//
// Reported twice — with corpus bigrams on and off.  The delta is the context
// contribution, which is the number the current benchmark suite cannot see.
//
// Usage:
//   cargo run --release --bin evaluate_sentences -- \
//     [--model data/akshar.model] [--corpus data/store/corpus_clean.txt] \
//     [--n 3000] [--topk 5] [--holdout-denom 200] [--show-misses 15]

use akshar_ime::core::akshara::segment;
use akshar_ime::core::engine::ImeEngine;
use akshar_ime::core::holdout::{is_holdout, DEFAULT_HOLDOUT_DENOM};
use akshar_ime::core::translit_model::TranslitModel;
use akshar_ime::core::unified::UnifiedModel;
use akshar_ime::fuzzy::grammar::orthographic_skeleton;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

/// Per-akshara best roman spelling, precomputed once from the EM emissions.
struct Romanizer {
    best: HashMap<String, String>,
}

impl Romanizer {
    fn new(model: &TranslitModel) -> Self {
        let mut best = HashMap::with_capacity(model.aksharas.len());
        for (aid, emissions) in model.emissions.iter().enumerate() {
            let Some(akshara) = model.aksharas.get(aid) else {
                continue;
            };
            // Emissions are -log P, so the minimum weight is the most likely
            // chunk.  Ties broken by the shorter, then lexicographically
            // smaller chunk to keep this deterministic across rebuilds.
            let pick = emissions
                .iter()
                .filter_map(|(cid, w)| model.chunks.get(*cid as usize).map(|c| (c, *w)))
                .filter(|(c, _)| !c.is_empty())
                .min_by(|(ca, wa), (cb, wb)| {
                    wa.total_cmp(wb)
                        .then(ca.len().cmp(&cb.len()))
                        .then(ca.cmp(cb))
                });
            if let Some((chunk, _)) = pick {
                best.insert(akshara.clone(), chunk.clone());
            }
        }
        Self { best }
    }

    /// Roman spelling of a Devanagari word, or None if any akshara is unknown
    /// to the model (such a word could never be typed into this engine, so
    /// scoring it would measure nothing).
    fn romanize(&self, word: &str) -> Option<String> {
        let mut out = String::with_capacity(word.len());
        let units = segment(word);
        if units.is_empty() {
            return None;
        }
        for unit in units {
            out.push_str(self.best.get(&unit)?);
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
}

#[derive(Default)]
struct Tally {
    words: usize,
    top1: usize,
    topk: usize,
    /// Sum of the gold word's 0-based rank, counted only over words where the
    /// gold appeared at all; `ranked` is that denominator.
    rank_sum: usize,
    ranked: usize,
    sentences: usize,
    sentences_exact: usize,
    /// Top-1 misses where the predicted word is orthographically equivalent
    /// to the gold under core Nepali spelling rules (hrasva/dirgha, श/ष/स,
    /// ब/व, nasals).  These are spelling-variant confusions, not decoding
    /// failures, and are the headroom available to the grammar engine.
    skeleton_misses: usize,
    /// Misses where *some* candidate in the returned list is skeleton-equal
    /// to the gold — the headroom for variant merging plus reranking.
    skeleton_in_list: usize,
    /// Of `skeleton_misses`, those where the predicted word romanizes to the
    /// SAME roman string as the gold.  These are genuinely ambiguous given
    /// the input: no orthographic rule can separate them, only context or
    /// user history can.  The remainder is the real grammar headroom.
    skeleton_misses_same_roman: usize,
}

impl Tally {
    fn pct(n: usize, d: usize) -> f64 {
        if d == 0 {
            0.0
        } else {
            n as f64 / d as f64 * 100.0
        }
    }

    fn report(&self, label: &str, k: usize) {
        println!(
            "  {label:<18} word@1={:>6.2}%  word@{k}={:>6.2}%  sentence-exact={:>6.2}%  \
             mean-gold-rank={:.2} (recall={:.2}%)\n  {:<18} spelling-variant top-1 misses={:>6.2}% of all words              ({} of {} misses), of which {} are input-ambiguous (same roman) \
             leaving {:.2}pp resolvable; variant present in list for {:>6.2}% of words",
            Self::pct(self.top1, self.words),
            Self::pct(self.topk, self.words),
            Self::pct(self.sentences_exact, self.sentences),
            if self.ranked == 0 {
                f64::NAN
            } else {
                self.rank_sum as f64 / self.ranked as f64
            },
            Self::pct(self.ranked, self.words),
            "",
            Self::pct(self.skeleton_misses, self.words),
            self.skeleton_misses,
            self.words - self.top1,
            self.skeleton_misses_same_roman,
            Self::pct(self.skeleton_misses - self.skeleton_misses_same_roman, self.words),
            Self::pct(self.skeleton_in_list, self.words),
        );
    }
}

fn main() {
    let mut model_path = "data/akshar.model".to_string();
    let mut corpus_path = "data/store/corpus_clean.txt".to_string();
    let mut n_sentences = 3000usize;
    let mut topk = 5usize;
    let mut holdout_denom = DEFAULT_HOLDOUT_DENOM;
    let mut show_misses = 0usize;
    // Sweep the corpus-bigram boost instead of a plain on/off run.  The
    // context contribution is a property of the fusion weight as much as of
    // the table, so a single on/off number cannot tell "the table is useless"
    // apart from "the weight is wrong".
    let mut boost_sweep: Vec<f64> = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model_path = args.next().expect("value for --model"),
            "--corpus" => corpus_path = args.next().expect("value for --corpus"),
            "--n" => n_sentences = args.next().expect("value for --n").parse().unwrap(),
            "--topk" => topk = args.next().expect("value for --topk").parse().unwrap(),
            "--holdout-denom" => {
                holdout_denom = args
                    .next()
                    .expect("value for --holdout-denom")
                    .parse()
                    .unwrap()
            }
            "--boost-sweep" => {
                boost_sweep = args
                    .next()
                    .expect("value for --boost-sweep")
                    .split(',')
                    .map(|v| v.trim().parse().expect("numeric boost"))
                    .collect();
            }
            "--show-misses" => {
                show_misses = args
                    .next()
                    .expect("value for --show-misses")
                    .parse()
                    .unwrap()
            }
            "--help" | "-h" => {
                println!("evaluate_sentences — in-context word accuracy on held-out sentences");
                println!("  --model <path>          unified model (default data/akshar.model)");
                println!("  --corpus <path>         corpus_clean.txt");
                println!("  --n <count>             held-out sentences to score (default 3000)");
                println!("  --topk <k>              top-k cutoff (default 5)");
                println!("  --holdout-denom <d>     hold out 1 sentence in d (default 200)");
                println!("  --boost-sweep <a,b,c>   sweep corpus-bigram boost values");
                println!("  --show-misses <n>       print n in-context top-1 misses");
                return;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    // ---------------------------------------------------------------------
    // Collect the held-out sentences first: this streams 1.4 GiB, so do it
    // before allocating the model.
    // ---------------------------------------------------------------------
    let t0 = Instant::now();
    let file = std::fs::File::open(&corpus_path).unwrap_or_else(|e| {
        eprintln!("cannot open corpus {corpus_path}: {e}");
        std::process::exit(1);
    });
    let mut sentences: Vec<Vec<String>> = Vec::with_capacity(n_sentences);
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if sentences.len() >= n_sentences {
            break;
        }
        let line = line.trim();
        if !is_holdout(line, holdout_denom) {
            continue;
        }
        let words: Vec<String> = line.split_whitespace().map(str::to_string).collect();
        // Single-word lines carry no context and would just re-measure the
        // isolated-word benchmark.
        if words.len() >= 2 {
            sentences.push(words);
        }
    }
    if sentences.is_empty() {
        eprintln!("no held-out sentences found (denom={holdout_denom}) — nothing to score");
        std::process::exit(1);
    }
    eprintln!(
        "Collected {} held-out sentences (1-in-{}) in {:.1?}",
        sentences.len(),
        holdout_denom,
        t0.elapsed()
    );

    // ---------------------------------------------------------------------
    // Load the model once; build the romanizer before handing it to the
    // engine so the 66 MB container is never cloned.
    // ---------------------------------------------------------------------
    let unified = UnifiedModel::load(Path::new(&model_path)).unwrap_or_else(|e| {
        eprintln!("cannot load model {model_path}: {e}");
        std::process::exit(1);
    });
    let romanizer = Romanizer::new(&unified.translit);
    eprintln!("Indexed best roman chunk for {} aksharas", romanizer.best.len());
    let mut engine = ImeEngine::from_unified(unified);

    // ---------------------------------------------------------------------
    // Score twice: context on, context off.  Same sentences, same romanization.
    // ---------------------------------------------------------------------
    let mut misses: Vec<(String, String, Vec<String>, String)> = Vec::new();
    let mut results = Vec::new();
    let mut skipped_words = 0usize;

    // (label, context enabled, boost).  A boost of None keeps the engine default.
    let mut runs: Vec<(String, bool, Option<f64>)> =
        vec![("context ON".to_string(), true, None), ("context OFF".to_string(), false, None)];
    for b in &boost_sweep {
        runs.push((format!("boost {b:>9.0}"), true, Some(*b)));
    }

    for (label, context_on, boost) in runs {
        engine.set_bigram_context_enabled(context_on);
        if let Some(b) = boost {
            engine.set_bigram_boost(b);
        }
        let context_on = context_on && boost.is_none();
        let mut t = Tally::default();
        let run_t0 = Instant::now();

        for words in &sentences {
            engine.set_context_word("");
            let mut sentence_perfect = true;
            let mut scored_any = false;

            for (i, gold) in words.iter().enumerate() {
                let Some(roman) = romanizer.romanize(gold) else {
                    if context_on {
                        skipped_words += 1;
                    }
                    // Unknown aksharas: cannot be typed, so it is not a miss.
                    // Still advance the context so later words see the truth.
                    engine.set_context_word(gold);
                    continue;
                };
                scored_any = true;
                t.words += 1;

                let suggestions = engine.get_suggestions(&roman, topk.max(8));
                let rank = suggestions.iter().position(|(dev, _)| dev == gold);

                if rank != Some(0) {
                    let gold_skel = orthographic_skeleton(gold);
                    if suggestions
                        .first()
                        .is_some_and(|(dev, _)| orthographic_skeleton(dev) == gold_skel)
                    {
                        t.skeleton_misses += 1;
                        // Would a user typing this word have produced the same
                        // keystrokes for the prediction?  If so the pair is
                        // unresolvable from the input alone.
                        if suggestions
                            .first()
                            .and_then(|(dev, _)| romanizer.romanize(dev))
                            .is_some_and(|r| r == roman)
                        {
                            t.skeleton_misses_same_roman += 1;
                        }
                    }
                    if suggestions
                        .iter()
                        .any(|(dev, _)| orthographic_skeleton(dev) == gold_skel)
                    {
                        t.skeleton_in_list += 1;
                    }
                }
                match rank {
                    Some(0) => {
                        t.top1 += 1;
                        t.topk += 1;
                        t.rank_sum += 0;
                        t.ranked += 1;
                    }
                    Some(r) => {
                        if r < topk {
                            t.topk += 1;
                        }
                        t.rank_sum += r;
                        t.ranked += 1;
                        sentence_perfect = false;
                    }
                    None => sentence_perfect = false,
                }

                if context_on && rank != Some(0) && misses.len() < show_misses {
                    misses.push((
                        roman.clone(),
                        gold.clone(),
                        suggestions.iter().take(4).map(|(d, _)| d.clone()).collect(),
                        words.get(i.wrapping_sub(1)).cloned().unwrap_or_default(),
                    ));
                }

                // Teacher forcing: the next word sees the gold context, not
                // our own possibly-wrong prediction.  Measures the bigram
                // table rather than compounding decode error.
                engine.set_context_word(gold);
            }

            if scored_any {
                t.sentences += 1;
                if sentence_perfect {
                    t.sentences_exact += 1;
                }
            }
        }
        results.push((label, t, run_t0.elapsed()));
    }

    // ---------------------------------------------------------------------
    println!(
        "\n=== Sentence-level evaluation ({} sentences, held out 1-in-{}) ===",
        sentences.len(),
        holdout_denom
    );
    println!("model: {model_path}");
    if skipped_words > 0 {
        println!("skipped {skipped_words} words with aksharas unknown to the model");
    }
    for (label, t, elapsed) in &results {
        t.report(label, topk);
        eprintln!("    ({} words in {:.1?})", t.words, elapsed);
    }
    {
        let on = &results[0].1;
        let off = &results[1].1;
        let d1 = Tally::pct(on.top1, on.words) - Tally::pct(off.top1, off.words);
        let dk = Tally::pct(on.topk, on.words) - Tally::pct(off.topk, off.words);
        let ds = Tally::pct(on.sentences_exact, on.sentences)
            - Tally::pct(off.sentences_exact, off.sentences);
        println!(
            "\n  CONTEXT CONTRIBUTION: word@1 {d1:+.2}pp   word@{topk} {dk:+.2}pp   \
             sentence-exact {ds:+.2}pp"
        );
    }

    if !misses.is_empty() {
        println!("\n  in-context top-1 misses (first {}):", misses.len());
        for (roman, gold, top, prev) in &misses {
            println!("    prev=`{prev}` roman=`{roman}` gold=`{gold}` top={top:?}");
        }
    }
}
