// Where does a keystroke's time actually go?
use akshar_ime::core::decoder::{DecoderConfig, ModelDecoder};
use akshar_ime::core::reranker::{rerank_with_table, FreqRanks};
use akshar_ime::core::unified::UnifiedModel;
use akshar_ime::ImeEngine;
use std::io::BufRead;
use std::time::Instant;

fn main() {
    let u = UnifiedModel::load(std::path::Path::new("data/akshar.model")).expect("load");
    let vocab = u.vocab_freq.clone();
    let ranks = FreqRanks::from_freq_map(&vocab);
    let trie = akshar_ime::core::wordtrie::WordTrie::from_freq_map(
        &vocab, &|a| u.translit.akshara_id(a), 1);
    let sparse = u.sparse_reranker_table.clone();
    let sscale = u.sparse_scale;
    let mut m = u.translit.clone();
    m.build_trigram_index();
    let dec = ModelDecoder::with_config(m, DecoderConfig::default());

    // Real queries, not synthetic ones: length distribution drives beam cost.
    let f = std::fs::File::open("data/aksharantar/test_devanagari.jsonl").unwrap();
    let mut qs: Vec<String> = Vec::new();
    for line in std::io::BufReader::new(f).lines().take(4000) {
        let l = line.unwrap();
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&l) {
            if let Some(r) = v["english word"].as_str() { qs.push(r.to_string()); }
        }
        if qs.len() >= 1000 { break; }
    }
    let n = qs.len() as f64;
    println!("{} queries, mean length {:.1} chars\n", qs.len(),
             qs.iter().map(|q| q.len()).sum::<usize>() as f64 / n);

    macro_rules! phase {
        ($label:expr, $body:expr) => {{
            let t = Instant::now();
            let r = $body;
            let ms = t.elapsed().as_secs_f64() * 1000.0 / n;
            println!("  {:<38} {:>7.3} ms/query", $label, ms);
            (r, ms)
        }};
    }

    let (_, t_free) = phase!("decode_detailed (free beam)", {
        let mut acc = 0usize;
        for q in &qs { acc += dec.decode_detailed(q, 50).len(); }
        acc
    });
    let (_, t_trie) = phase!("decode_in_words_detailed (trie beam)", {
        let mut acc = 0usize;
        for q in &qs { acc += dec.decode_in_words_detailed(q, 50, &trie).len(); }
        acc
    });
    let (cands, t_union) = phase!("decode_union (both + merge)", {
        let mut v = Vec::new();
        for q in &qs { v.push(dec.decode_union(q, 50, Some(&trie))); }
        v
    });
    let (_, t_rr) = phase!("rerank_with_table", {
        let mut acc = 0usize;
        for (q, c) in qs.iter().zip(&cands) {
            acc += rerank_with_table(q, c, &vocab, &ranks, Some(&sparse), Some(sscale)).len();
        }
        acc
    });

    let engine = ImeEngine::new();
    let (_, t_all) = phase!("ImeEngine::get_suggestions (end to end)", {
        let mut acc = 0usize;
        for q in &qs { acc += engine.get_suggestions(q, 5).len(); }
        acc
    });

    println!("\n  decode_union share of end-to-end : {:.0}%", t_union / t_all * 100.0);
    println!("  rerank      share of end-to-end : {:.0}%", t_rr / t_all * 100.0);
    println!("  free beam vs trie beam          : {:.2} / {:.2} ms", t_free, t_trie);
    println!("  unaccounted (engine overhead)   : {:.3} ms", (t_all - t_union - t_rr).max(0.0));
}
