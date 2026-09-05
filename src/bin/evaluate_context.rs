// File: src/bin/evaluate_context.rs
//
// E6 measurement harness: how much does the corpus-bigram context boost
// (previous committed word) improve suggestion accuracy on real sentences?
//
// Protocol: read sentences from the cleaned corpus, romanize every word with
// the model's own emission argmax (the canonical spelling a user would type),
// then walk two identical engines in lockstep:
//   A) context disabled  B) context enabled (bigram boost)
// Both engines receive the same confirmations (identical learning state);
// the ONLY difference is the corpus-bigram boost. Words after the first in a
// sentence are the context-applicable subset.
//
// Usage:
//   cargo run --release --bin evaluate_context -- \
//     [--sentences 1000] [--skip 0] [--boost 20000] [--topk 8]
use akshar_ime::ImeEngine;
use std::io::BufRead;

/// Canonical roman spelling of a word under the model's emission argmax.
fn romanize(engine: &ImeEngine, word: &str) -> Option<String> {
    let aksharas = akshar_ime::core::akshara::segment(word);
    if aksharas.is_empty() {
        return None;
    }
    let mut out = String::new();
    for a in &aksharas {
        let id = engine.decoder.model.akshara_id(a)?;
        let (chunk, _) = engine.decoder.model.top_emissions(id, 1).into_iter().next()?;
        out.push_str(&chunk);
    }
    Some(out)
}

#[derive(Default)]
struct Acc {
    hit: usize,
    total: usize,
}

impl Acc {
    fn add(&mut self, ok: bool) {
        self.total += 1;
        self.hit += ok as usize;
    }
    fn pct(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.hit as f64 / self.total as f64 * 100.0
        }
    }
}

fn main() {
    let mut sentences = 1000usize;
    let mut skip = 0usize;
    let mut boost = 20_000.0f64;
    let mut topk = 8usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--sentences" => sentences = args.next().expect("n").parse().unwrap(),
            "--skip" => skip = args.next().expect("n").parse().unwrap(),
            "--boost" => boost = args.next().expect("x").parse().unwrap(),
            "--topk" => topk = args.next().expect("k").parse().unwrap(),
            other => panic!("unknown arg {other}"),
        }
    }

    // Engine A: context off. Engine B: context on.
    let mut a = ImeEngine::new();
    let mut b = ImeEngine::new();
    a.set_bigram_context_enabled(false);
    b.set_bigram_context_enabled(true);
    b.set_bigram_boost(boost);

    let file = std::fs::File::open("data/store/corpus_clean.txt").expect("corpus");
    let reader = std::io::BufReader::new(file);

    let mut acc_a = Acc::default();
    let mut acc_b = Acc::default();
    let mut ctx_acc_a = Acc::default();
    let mut ctx_acc_b = Acc::default();
    let mut n_seen = 0usize;

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        let words: Vec<&str> = line.split_whitespace().collect();
        if words.len() < 3 {
            continue;
        }
        let romans: Vec<Option<String>> = words.iter().map(|w| romanize(&a, w)).collect();
        if romans.iter().any(|r| r.is_none()) {
            continue; // keep both engines' learning states in lockstep
        }
        if n_seen < skip {
            n_seen += 1;
            continue;
        }
        n_seen += 1;
        if n_seen - skip > sentences {
            break;
        }
        for (i, (word, roman)) in words.iter().zip(romans.iter()).enumerate() {
            let roman = roman.as_ref().unwrap();
            let sa = a.get_suggestions(roman, topk);
            let sb = b.get_suggestions(roman, topk);
            let ok_a = sa.first().is_some_and(|(d, _)| d == word);
            let ok_b = sb.first().is_some_and(|(d, _)| d == word);
            acc_a.add(ok_a);
            acc_b.add(ok_b);
            if i > 0 {
                ctx_acc_a.add(ok_a);
                ctx_acc_b.add(ok_b);
            }
            // identical confirmations keep learning + history in lockstep
            a.user_confirms(roman, word);
            b.user_confirms(roman, word);
        }
    }

    println!(
        "sentences {} (skip {}), boost {:.0}, topk {}",
        sentences.min(n_seen.saturating_sub(skip)),
        skip,
        boost,
        topk
    );
    println!(
        "A context OFF: all {}/{} = {:.2}%   | post-first-word {}/{} = {:.2}%",
        acc_a.hit, acc_a.total, acc_a.pct(),
        ctx_acc_a.hit, ctx_acc_a.total, ctx_acc_a.pct()
    );
    println!(
        "B context ON : all {}/{} = {:.2}%   | post-first-word {}/{} = {:.2}%",
        acc_b.hit, acc_b.total, acc_b.pct(),
        ctx_acc_b.hit, ctx_acc_b.total, ctx_acc_b.pct()
    );
}
