// File: src/bin/parity_v2.rs
//
// One-shot validation that the Rust port of reranker v2 reproduces the
// Python model's rankings exactly on exported parity cases
// (data/pipeline/reranker_v2.py::export_rust -> /tmp/v2_parity.json).
//
// Usage: cargo run --release --bin parity_v2 -- /tmp/v2_parity.json
use akshar_ime::core::decoder::DecodedCandidate;
use akshar_ime::core::reranker_v2::{rerank, RerankerV2Data};
use std::collections::HashMap;

fn main() {
    let path = std::env::args().nth(1).unwrap_or("/tmp/v2_parity.json".into());
    let spec: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(path).expect("parity json")).expect("json");

    // The real corpus vocabulary — same file the Python side ranked with.
    let bytes = std::fs::read("data/word_freq_text.bin").expect("vocab bin");
    let v2 = RerankerV2Data::from_bin_bytes(&bytes).expect("vocab parse");

    let mut pass = 0;
    let cases = spec["cases"].as_array().expect("cases");
    for (i, case) in cases.iter().enumerate() {
        let roman = case["roman"].as_str().unwrap();
        let cands: Vec<DecodedCandidate> = case["cands"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| DecodedCandidate {
                dev: c[0].as_str().unwrap().to_string(),
                emit: c[1].as_f64().unwrap(),
                lm: c[2].as_f64().unwrap(),
                akshara_count: c[3].as_u64().unwrap() as usize,
            })
            .collect();
        let got: Vec<String> = rerank(roman, &cands, &v2.freq, &v2.ranks)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        let want: Vec<String> = case["expected_top5"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_string())
            .collect();
        let ok = &got[..want.len().min(got.len())] == want.as_slice();
        if ok {
            pass += 1;
        } else {
            println!("case {i} MISMATCH\n  want {want:?}\n  got  {:?}", &got[..want.len().min(got.len())]);
        }
    }
    println!("parity: {pass}/{len} cases match", len = cases.len());
    if pass != cases.len() {
        std::process::exit(1);
    }
}
