// File: src/bin/build/pack_model.rs
//
// Pack individual model artifacts:
//   1. Transliteration model (translit_model.bin)
//   2. Sparse reranker weights (reranker_weights_sparse.bin)
//   3. Text word frequency vocabulary (word_freq_text.bin)
//   4. (Optional) Word bigrams (word_bigrams.bin)
// into a single unified `akshar.model` file.
//
// Usage: cargo run --release --bin pack_model -- [options]
//   --translit <path>  (default: data/translit_model.bin)
//   --sparse <path>    (default: data/reranker_weights_sparse.bin)
//   --vocab <path>     (default: data/word_freq_text.bin)
//   --bigrams <path>   (default: data/word_bigrams.bin if present)
//   --out <path>       (default: data/akshar.model)
//   --min-freq <n>     (optional prune threshold for vocab, default: 1)

use akshar_ime::core::translit_model::TranslitModel;
use akshar_ime::core::unified::UnifiedModel;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let mut translit_p = PathBuf::from("data/translit_model.bin");
    let mut sparse_p = PathBuf::from("data/reranker_weights_sparse.bin");
    let mut vocab_p = PathBuf::from("data/word_freq_text.bin");
    let mut bigrams_p: Option<PathBuf> = {
        let p = PathBuf::from("data/word_bigrams.bin");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    };
    let mut out_p = PathBuf::from("data/akshar.model");
    let mut min_freq: u32 = 1;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--translit" => translit_p = PathBuf::from(args.next().expect("value for --translit")),
            "--sparse" => sparse_p = PathBuf::from(args.next().expect("value for --sparse")),
            "--vocab" => vocab_p = PathBuf::from(args.next().expect("value for --vocab")),
            "--bigrams" => bigrams_p = Some(PathBuf::from(args.next().expect("value for --bigrams"))),
            "--no-bigrams" => bigrams_p = None,
            "--out" | "-o" => out_p = PathBuf::from(args.next().expect("value for --out")),
            "--min-freq" => min_freq = args.next().expect("value for --min-freq").parse().unwrap(),
            "-h" | "--help" => {
                println!("Usage: cargo run --release --bin pack_model -- [options]");
                println!("  --translit <path>  EM transliteration model");
                println!("  --sparse <path>    Sparse reranker table");
                println!("  --vocab <path>     Word frequency vocabulary");
                println!("  --bigrams <path>   Word bigrams (optional)");
                println!("  --no-bigrams       Exclude bigrams");
                println!("  --min-freq <n>     Prune vocabulary below this frequency (default: 1)");
                println!("  --out <path>       Output path (default: data/akshar.model)");
                return;
            }
            other => {
                eprintln!("Unknown option: {other}");
                std::process::exit(2);
            }
        }
    }

    println!("============================================================");
    println!("             Akshar Unified Model Packer                    ");
    println!("============================================================");

    let t0 = Instant::now();

    print!("1. Loading transliteration model ({}) ... ", translit_p.display());
    let translit = TranslitModel::load(&translit_p).expect("load translit model");
    println!("done ({} aksharas, {} chunks)", translit.aksharas.len(), translit.chunks.len());

    print!("2. Loading sparse reranker table ({}) ... ", sparse_p.display());
    let sparse_bytes = std::fs::read(&sparse_p).expect("read sparse table");
    let sparse_table: Vec<i8> = sparse_bytes.into_iter().map(|b| b as i8).collect();
    println!("done ({} weights)", sparse_table.len());

    print!("3. Loading word frequency vocabulary ({}) ... ", vocab_p.display());
    let vocab_bytes = std::fs::read(&vocab_p).expect("read vocab");
    let mut vocab: HashMap<String, u32> = bincode::deserialize(&vocab_bytes).expect("deserialize vocab");
    let initial_vocab_len = vocab.len();
    if min_freq > 1 {
        vocab.retain(|_, &mut c| c >= min_freq);
    }
    println!("done ({} words retained from {})", vocab.len(), initial_vocab_len);

    let bigrams = if let Some(ref bp) = bigrams_p {
        if bp.exists() {
            print!("4. Loading word bigrams ({}) ... ", bp.display());
            let bigram_bytes = std::fs::read(bp).expect("read bigrams");
            let bg: HashMap<String, Vec<(String, u32)>> = bincode::deserialize(&bigram_bytes).expect("deserialize bigrams");
            println!("done ({} context heads)", bg.len());
            Some(bg)
        } else {
            println!("4. Word bigrams file not found; omitting.");
            None
        }
    } else {
        println!("4. Skipping word bigrams (not requested).");
        None
    };

    println!("5. Assembling and writing unified model -> {} ...", out_p.display());
    let unified = UnifiedModel::new(translit, sparse_table, vocab, bigrams);
    unified.save(&out_p).expect("save unified model");

    let metadata = std::fs::metadata(&out_p).expect("metadata");
    let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);

    println!("============================================================");
    println!("Successfully packed unified model in {:.2?}!", t0.elapsed());
    println!("Artifact: {} ({:.2} MB)", out_p.display(), size_mb);
    println!("============================================================");
}
