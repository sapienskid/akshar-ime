// File: src/bin/build/repack_model.rs
//
// Read any container version and write it back in the current one.

use akshar_ime::core::unified::UnifiedModel;
use std::path::Path;

/// Renumber the akshara table down to the entries that are actually used.
fn compact_aksharas(m: &mut UnifiedModel) -> (usize, usize) {
    let t = &mut m.translit;
    let n = t.aksharas.len();
    let mut keep = vec![false; n];

    for (a, row) in t.emissions.iter().enumerate() {
        if !row.is_empty() {
            keep[a] = true;
        }
    }
    for (a, row) in t.bigrams.iter().enumerate() {
        if !row.is_empty() {
            keep[a] = true;
            for &(b, _) in row {
                if let Some(k) = keep.get_mut(b as usize) {
                    *k = true;
                }
            }
        }
    }
    for (i, &(a, b)) in t.trigram_keys.iter().enumerate() {
        if t.trigrams.get(i).is_some_and(|r| !r.is_empty()) {
            for id in [a, b] {
                if let Some(k) = keep.get_mut(id as usize) {
                    *k = true;
                }
            }
            for &(c, _) in &t.trigrams[i] {
                if let Some(k) = keep.get_mut(c as usize) {
                    *k = true;
                }
            }
        }
    }

    let mut remap = vec![u32::MAX; n];
    let mut next = 0u32;
    for (old, &k) in keep.iter().enumerate() {
        if k {
            remap[old] = next;
            next += 1;
        }
    }
    let kept = next as usize;
    if kept == n {
        return (n, n);
    }

    let reindex = |v: &mut Vec<Vec<(u32, f32)>>, remap: &[u32], keep: &[bool]| {
        let mut it = keep.iter();
        v.retain(|_| *it.next().unwrap_or(&false));
        for row in v.iter_mut() {
            row.retain(|&(id, _)| remap.get(id as usize).is_some_and(|&r| r != u32::MAX));
            for e in row.iter_mut() {
                e.0 = remap[e.0 as usize];
            }
        }
    };

    let mut it = keep.iter();
    t.emissions.retain(|_| *it.next().unwrap_or(&false));

    reindex(&mut t.bigrams, &remap, &keep);

    for arr in [&mut t.backoff, &mut t.unigram_kn, &mut t.word_start] {
        let mut it = keep.iter();
        arr.retain(|_| *it.next().unwrap_or(&false));
    }

    let ctx_keep: Vec<bool> = t
        .trigram_keys
        .iter()
        .map(|&(a, b)| {
            remap.get(a as usize).is_some_and(|&r| r != u32::MAX)
                && remap.get(b as usize).is_some_and(|&r| r != u32::MAX)
        })
        .collect();
    let mut it = ctx_keep.iter();
    t.trigrams.retain(|_| *it.next().unwrap());
    let mut it = ctx_keep.iter();
    t.trigram_backoff.retain(|_| *it.next().unwrap());
    let mut it = ctx_keep.iter();
    t.trigram_keys.retain(|_| *it.next().unwrap());

    for key in t.trigram_keys.iter_mut() {
        key.0 = remap[key.0 as usize];
        key.1 = remap[key.1 as usize];
    }
    for row in t.trigrams.iter_mut() {
        row.retain(|&(c, _)| remap.get(c as usize).is_some_and(|&r| r != u32::MAX));
        for e in row.iter_mut() {
            e.0 = remap[e.0 as usize];
        }
    }

    let mut it = keep.iter();
    t.aksharas.retain(|_| *it.next().unwrap_or(&false));

    t.build_trigram_index();
    (n, kept)
}

fn main() {
    let mut model_path = "data/akshar.model".to_string();
    let mut out_path = "data/akshar.model".to_string();
    let mut compact = false;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model_path = args.next().expect("value for --model"),
            "--out" => out_path = args.next().expect("value for --out"),
            "--compact-aksharas" => compact = true,
            "--help" | "-h" => {
                println!("repack_model — rewrite a model in the current container format");
                println!("  --model <path>   input (any container version)");
                println!("  --out <path>     output (current version)");
                println!("  --compact-aksharas  renumber the akshara table to used entries");
                return;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    let before = std::fs::metadata(&model_path).expect("stat input").len();
    let mut m = UnifiedModel::load(Path::new(&model_path)).expect("load model");
    println!("Loaded {model_path} ({:.2} MB)", before as f64 / 1048576.0);

    if compact {
        let (before_n, after_n) = compact_aksharas(&mut m);
        println!("Compacted aksharas: {before_n} -> {after_n}");
    }

    m.save(Path::new(&out_path)).expect("save model");
    let after = std::fs::metadata(&out_path).expect("stat output").len();
    println!(
        "Wrote {out_path} ({:.2} MB) — {:.1}% of the original, {:.2} MB saved",
        after as f64 / 1048576.0,
        after as f64 / before as f64 * 100.0,
        (before as f64 - after as f64) / 1048576.0
    );
}
