// File: src/bin/build_lm_from_text.rs
//
// M4-real experiment: rebuild the akshara Kneser-Ney LM from REAL running
// text (frequency-weighted wiki words) instead of the unique-heavy mined
// pairs, keeping the original EM emissions.  Hypothesis: the plateau in
// isolated-word ranking is an LM-data problem, not an emission problem.
//
// Usage: build_lm_from_text [--model data/translit_model.bin] \
//          [--vocab data/word_freq_text.bin] [--out data/translit_model_wiki.bin]
//          [--min-count 1] [--mix-a 0.0]  (0 = replace LM, else interpolate)

use akshar_ime::core::akshara::segment;
use akshar_ime::core::translit_model::TranslitModel;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::path::Path;

struct Counts {
    unigram: Vec<u64>,
    bigram: HashMap<(u32, u32), u64>,
    trigram: HashMap<(u32, u32, u32), u64>,
    word_initial: Vec<u64>,
    total_words: u64,
    continuation: Vec<u64>,
    distinct_bigrams: u64,
    trigram_successors: HashMap<(u32, u32), u64>,
    n: usize,
}

fn main() {
    let mut model_path = "data/translit_model.bin".to_string();
    let mut vocab_path = "data/word_freq_text.bin".to_string();
    let mut out_path = "data/translit_model_wiki.bin".to_string();
    let mut min_count = 1u32;
    let mut alpha = 1.0f64;
    let mut pair_paths: Vec<String> = Vec::new();
    let mut wiki_weight: u64 = 1;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_path = args.next().expect("p"),
            "--vocab" => vocab_path = args.next().expect("p"),
            "--out" => out_path = args.next().expect("p"),
            "--min-count" => min_count = args.next().expect("n").parse().expect("n"),
            "--alpha" => alpha = args.next().expect("f").parse().expect("f"),
            "--pairs" => pair_paths.push(args.next().expect("p")),
            "--wiki-weight" => wiki_weight = args.next().expect("f").parse().expect("f"),
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    let mut model = TranslitModel::load(Path::new(&model_path)).expect("load model");
    let n = model.aksharas.len();
    let id_of: HashMap<&str, u32> = model
        .aksharas
        .iter()
        .enumerate()
        .map(|(i, a)| (a.as_str(), i as u32))
        .collect();

    let map: HashMap<String, u32> =
        bincode::deserialize(&std::fs::read(&vocab_path).expect("vocab")).expect("deserialize");
    eprintln!("vocab entries: {} (min count {min_count})", map.len());

    let mut c = Counts {
        unigram: vec![0; n],
        bigram: HashMap::new(),
        trigram: HashMap::new(),
        word_initial: vec![0; n],
        total_words: 0,
        continuation: vec![0; n],
        distinct_bigrams: 0,
        trigram_successors: HashMap::new(),
        n,
    };

    let mut used_words = 0u64;
    // Optional native-side ingestion of the pair corpus (count-level mix).
    #[derive(serde::Deserialize)]
    struct Rec<'a> {
        #[serde(rename = "native word")]
        native: &'a str,
    }
    for pp in &pair_paths {
        let f = std::fs::File::open(pp).expect("pairs");
        for line in std::io::BufReader::new(f).lines().map_while(Result::ok) {
            let t = line.trim();
            if t.is_empty() { continue; }
            if let Ok(rec) = serde_json::from_str::<Rec>(t) {
                let mut aks = Vec::with_capacity(8);
                let mut ok = true;
                for unit in segment(rec.native.trim()) {
                    match id_of.get(unit.as_str()) {
                        Some(&id) => aks.push(id),
                        None => { ok = false; break; }
                    }
                }
                if ok && !aks.is_empty() {
                    used_words += 1;
                    for &a in &aks { c.unigram[a as usize] += 1; }
                    c.word_initial[aks[0] as usize] += 1;
                    c.total_words += 1;
                    for w in aks.windows(2) {
                        let (b, cc) = (w[0], w[1]);
                        let e = c.bigram.entry((b, cc)).or_insert(0);
                        if *e == 0 { c.continuation[cc as usize] += 1; c.distinct_bigrams += 1; }
                        *e += 1;
                    }
                    for w in aks.windows(3) {
                        let (a, b, cc) = (w[0], w[1], w[2]);
                        let e = c.trigram.entry((a, b, cc)).or_insert(0);
                        if *e == 0 { *c.trigram_successors.entry((a, b)).or_insert(0) += 1; }
                        *e += 1;
                    }
                }
            }
        }
    }
    let corpus_instances = used_words;
    for (word, &count) in &map {
        if count < min_count {
            continue;
        }
        let mut aks = Vec::with_capacity(8);
        let mut ok = true;
        for unit in segment(word) {
            match id_of.get(unit.as_str()) {
                Some(&id) => aks.push(id),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok || aks.is_empty() {
            continue;
        }
        used_words += count as u64 * wiki_weight;
        let count = count.saturating_mul(wiki_weight as u32);
        for &a in &aks {
            c.unigram[a as usize] += count as u64;
        }
        c.word_initial[aks[0] as usize] += count as u64;
        c.total_words += count as u64;
        for w in aks.windows(2) {
            let (b, cc) = (w[0], w[1]);
            let e = c.bigram.entry((b, cc)).or_insert(0);
            if *e == 0 {
                c.continuation[cc as usize] += 1;
                c.distinct_bigrams += 1;
            }
            *e += count as u64;
        }
        for w in aks.windows(3) {
            let (a, b, cc) = (w[0], w[1], w[2]);
            let e = c.trigram.entry((a, b, cc)).or_insert(0);
            if *e == 0 {
                *c.trigram_successors.entry((a, b)).or_insert(0) += 1;
            }
            *e += count as u64;
        }
    }
    eprintln!(
        "counted {used_words} word-instances (corpus {corpus_instances}) over {} akshara types; {} bigrams, {} trigrams",
        c.bigram.len(),
        c.trigram.len(),
        c.unigram.iter().filter(|&&x| x > 0).count()
    );

    // --- KN tables, mirroring em_trainer::build_kn_lm. ---
    let delta = 0.75f64;
    let distinct = c.distinct_bigrams as f64;
    let mut unigram_kn = vec![0.0f32; n];
    for a in 0..n {
        let cont = c.continuation[a] as f64;
        let p = (cont + 0.5) / (distinct + 0.5 * n as f64);
        unigram_kn[a] = -p.ln() as f32;
    }
    let mut by_left: HashMap<u32, Vec<(u32, u64)>> = HashMap::new();
    for (&(b, cc), &cnt) in &c.bigram {
        by_left.entry(b).or_default().push((cc, cnt));
    }
    let mut bigrams = vec![Vec::new(); n];
    let mut backoff = vec![0.0f32; n];
    for a in 0..n {
        let Some(list) = by_left.get(&(a as u32)) else {
            continue;
        };
        let total: u64 = list.iter().map(|(_, cnt)| cnt).sum();
        let n_distinct = list.len() as f64;
        let lambda = delta * n_distinct / total as f64;
        backoff[a] = (-lambda.ln()) as f32;
        let mut v = Vec::with_capacity(list.len());
        for &(cc, cnt) in list {
            let disc = (cnt as f64 - delta).max(0.0) / total as f64;
            let p_kn_c = (-unigram_kn[cc as usize] as f64).exp();
            let p = disc + lambda * p_kn_c;
            let w = if p > 0.0 { -p.ln() } else { 50.0 };
            v.push((cc, w as f32));
        }
        v.sort_by_key(|(id, _)| *id);
        bigrams[a] = v;
    }
    let mut by_ctx: HashMap<(u32, u32), Vec<(u32, u64)>> = HashMap::new();
    for (&(a, b, cc), &cnt) in &c.trigram {
        by_ctx.entry((a, b)).or_default().push((cc, cnt));
    }
    let mut ctxs: Vec<(u32, u32)> = by_ctx.keys().copied().collect();
    ctxs.sort();
    let mut trigram_keys = Vec::with_capacity(ctxs.len());
    let mut trigrams = Vec::with_capacity(ctxs.len());
    let mut trigram_backoff = Vec::with_capacity(ctxs.len());
    for &(a, b) in &ctxs {
        let list = &by_ctx[&(a, b)];
        let c_ab = c.bigram.get(&(a, b)).copied().unwrap_or(1) as f64;
        let a_succ = c.trigram_successors.get(&(a, b)).copied().unwrap_or(1) as f64;
        let lambda = delta * a_succ / c_ab;
        trigram_backoff.push((-lambda.ln()) as f32);
        let mut v = Vec::with_capacity(list.len());
        for &(cc, cnt) in list {
            let disc = (cnt as f64 - delta).max(0.0) / c_ab;
            let p_kn_c = (-model.bigram_weight(b, cc)).exp();
            let p = disc + lambda * p_kn_c;
            let w = if p > 0.0 { -p.ln() } else { 50.0 };
            v.push((cc, w as f32));
        }
        v.sort_by_key(|(id, _)| *id);
        trigram_keys.push((a, b));
        trigrams.push(v);
    }
    let mut word_start: Vec<f32> = c
        .word_initial
        .iter()
        .map(|&cnt| {
            let p = (cnt as f64 + 0.5) / (c.total_words as f64 + 0.5 * n as f64);
            -p.ln() as f32
        })
        .collect();

    // Interpolate with the base model's tables: p = a*p_wiki + (1-a)*p_base.
    if alpha < 1.0 {
        let blend_map = |wiki: Vec<Vec<(u32, f32)>>, base: Vec<Vec<(u32, f32)>>| -> Vec<Vec<(u32, f32)>> {
            wiki.iter().zip(base.iter()).map(|(wv, bv)| {
                let mut acc: HashMap<u32, f64> = HashMap::new();
                for (id, w) in bv { *acc.entry(*id).or_insert(0.0) += (1.0 - alpha) * (-*w as f64).exp(); }
                for (id, w) in wv { *acc.entry(*id).or_insert(0.0) += alpha * (-*w as f64).exp(); }
                let mut v: Vec<(u32, f32)> = acc.into_iter().map(|(id, p)| (id, -p.ln() as f32)).collect();
                v.sort_by_key(|(id, _)| *id);
                v
            }).collect()
        };
        bigrams = blend_map(bigrams, std::mem::take(&mut model.bigrams));
        trigrams = blend_map(trigrams, std::mem::take(&mut model.trigrams));
        let blend_vec = |w: Vec<f32>, b: Vec<f32>| -> Vec<f32> {
            w.iter().zip(b.iter()).map(|(&x, &y)| {
                let p = alpha * (-x as f64).exp() + (1.0 - alpha) * (-y as f64).exp();
                -p.ln() as f32
            }).collect()
        };
        unigram_kn = blend_vec(unigram_kn, std::mem::take(&mut model.unigram_kn));
        word_start = blend_vec(word_start, std::mem::take(&mut model.word_start));
        eprintln!("interpolated with alpha={alpha}");
    }
    model.bigrams = bigrams;
    model.backoff = backoff;
    model.unigram_kn = unigram_kn;
    model.word_start = word_start;
    model.trigram_keys = trigram_keys;
    model.trigrams = trigrams;
    model.trigram_backoff = trigram_backoff;
    model.build_trigram_index();
    assert!(model.validate(), "hybrid model failed validation");
    model.save(Path::new(&out_path)).expect("save");
    eprintln!("saved hybrid LM model -> {out_path}");
    let _ = HashSet::<u32>::new(); // silence unused import if refactored
}
