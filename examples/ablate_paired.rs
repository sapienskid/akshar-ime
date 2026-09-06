// Emit per-case correctness so ablations can be compared with a PAIRED test.
//
// Comparing two independent accuracy numbers and their CIs understates the
// evidence: the two systems see the same cases, so what matters is the cases
// where they disagree (McNemar), not the marginal totals.
use akshar_ime::ImeEngine;
use std::io::BufRead;

fn main() {
    let engine = ImeEngine::new();
    let f = std::fs::File::open("data/aksharantar/test_devanagari.jsonl").expect("test split");
    for line in std::io::BufReader::new(f).lines() {
        let Ok(l) = line else { break };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&l) else {
            continue;
        };
        let (Some(roman), Some(gold), Some(src)) = (
            v["english word"].as_str(),
            v["native word"].as_str(),
            v["source"].as_str(),
        ) else {
            continue;
        };
        let s = engine.get_suggestions(roman, 5);
        let t1 = s.first().is_some_and(|(d, _)| d == gold) as u8;
        let t5 = s.iter().any(|(d, _)| d == gold) as u8;
        println!("{src}\t{t1}\t{t5}");
    }
}
