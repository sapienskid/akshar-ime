// File: src/bin/build/pack_model.rs
//
// Pack individual model artifacts:
//   1. Transliteration model (translit_model.bin)
//   2. Sparse reranker weights (reranker_weights_sparse.bin)
//   3. Text word frequency vocabulary (word_freq_text.bin)
// into a single unified `akshar.model` file.

use akshar_ime::core::translit_model::TranslitModel;
use akshar_ime::core::unified::UnifiedModel;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let mut translit_p = PathBuf::from("data/translit_model.bin");
    let mut sparse_p = PathBuf::from("data/reranker_weights_sparse.bin");
    let mut vocab_p = PathBuf::from("data/word_freq_text.bin");
    let mut out_p = PathBuf::from("data/akshar.model");
    let mut min_freq: u32 = 1;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--translit" => translit_p = PathBuf::from(args.next().expect("value for --translit")),
            "--sparse" => sparse_p = PathBuf::from(args.next().expect("value for --sparse")),
            "--vocab" => vocab_p = PathBuf::from(args.next().expect("value for --vocab")),
            "--out" | "-o" => out_p = PathBuf::from(args.next().expect("value for --out")),
            "--min-freq" => min_freq = args.next().expect("value for --min-freq").parse().unwrap(),
            "-h" | "--help" => {
                println!("Usage: cargo run --release --bin pack_model -- [options]");
                println!("  --translit <path>         EM transliteration model");
                println!("  --sparse <path>           Sparse reranker table");
                println!("  --vocab <path>            Word frequency vocabulary");
                println!("  --min-freq <n>            Prune vocabulary below this frequency (default: 1)");
                println!("  --out <path>              Output path (default: data/akshar.model)");
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

    print!(
        "1. Loading transliteration model ({}) ... ",
        translit_p.display()
    );
    let translit = TranslitModel::load(&translit_p).expect("load translit model");
    println!(
        "done ({} aksharas, {} chunks)",
        translit.aksharas.len(),
        translit.chunks.len()
    );

    print!(
        "2. Loading sparse reranker table ({}) ... ",
        sparse_p.display()
    );
    let sparse_bytes = std::fs::read(&sparse_p).expect("read sparse table");
    let sparse_table: Vec<i8> = sparse_bytes.into_iter().map(|b| b as i8).collect();
    println!("done ({} weights)", sparse_table.len());

    print!(
        "3. Loading word frequency vocabulary ({}) ... ",
        vocab_p.display()
    );
    let vocab_bytes = std::fs::read(&vocab_p).expect("read vocab");
    let mut vocab: HashMap<String, u32> =
        bincode::deserialize(&vocab_bytes).expect("deserialize vocab");
    let initial_vocab_len = vocab.len();
    if min_freq > 1 {
        vocab.retain(|_, &mut c| c >= min_freq);
    }
    println!(
        "done ({} words retained from {})",
        vocab.len(),
        initial_vocab_len
    );

    println!(
        "4. Assembling and writing unified model -> {} ...",
        out_p.display()
    );
    let unified = UnifiedModel::new(
        translit,
        sparse_table,
        akshar_ime::core::reranker_weights::SPARSE_SCALE,
        vocab,
    );
    unified.save(&out_p).expect("save unified model");

    let metadata = std::fs::metadata(&out_p).expect("metadata");
    let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);

    println!("============================================================");
    println!("Successfully packed unified model in {:.2?}!", t0.elapsed());
    println!("Artifact: {} ({:.2} MB)", out_p.display(), size_mb);
    println!("============================================================");
}
