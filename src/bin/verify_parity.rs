// File: src/bin/verify_parity.rs
//
// One-shot validation that the Rust port of the discriminative reranker reproduces the
// exported parity cases exactly.
//
// Usage: cargo run --release --bin verify_parity -- /tmp/parity.json
use akshar_ime::core::decoder::DecodedCandidate;
use akshar_ime::core::reranker::{rerank, RerankerData};

fn main() {
    let path = std::env::args().nth(1).unwrap_or("/tmp/parity.json".into());
    let spec: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(path).expect("parity json")).expect("json");

    // The real corpus vocabulary.
    let bytes = std::fs::read("data/word_freq_text.bin").expect("vocab bin");
    let reranker_data = RerankerData::from_bin_bytes(&bytes).expect("vocab parse");

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
        let got: Vec<String> = rerank(roman, &cands, &reranker_data.freq, &reranker_data.ranks)
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
