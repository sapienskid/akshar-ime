// File: build.rs
//
// `data/` is entirely gitignored (see .gitignore and data/README.md): model
// artifacts are trained locally and attached to releases separately via
// `make release-upload`, never committed. But
// `src/core/reranker_weights.rs` used `include_bytes!("../../data/reranker_weights_sparse.bin")`
// directly, which made that file a *compile-time* dependency of the crate.
// A fresh clone -- including CI -- has no such file and cannot compile at
// all.
//
// This script copies the file into OUT_DIR when it exists (so a local dev
// build with a trained model embeds the real legacy weights, unchanged
// behavior), and writes an empty placeholder when it does not, so the crate
// always compiles. The embedded table is only a last-resort fallback: the
// unified model container carries its own sparse table via
// `UnifiedModel::sparse_reranker_table`, which is what `ImeEngine::from_unified`
// actually uses (see `reranker::rerank_with_table`'s `custom_sparse_table`).

use std::path::Path;

fn main() {
    let src = Path::new("data/reranker_weights_sparse.bin");
    println!("cargo:rerun-if-changed={}", src.display());

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let dest = Path::new(&out_dir).join("reranker_weights_sparse.bin");

    if src.exists() {
        std::fs::copy(src, &dest).expect("copy reranker_weights_sparse.bin into OUT_DIR");
    } else {
        std::fs::write(&dest, []).expect("write empty reranker_weights_sparse.bin placeholder");
    }
}
