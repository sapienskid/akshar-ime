// Minimal test harness for Nepali IME RomanizationEngine
// Run with: cargo run --bin ime_test
// src/bin/ime_test.rs
use nepali_smart_ime::core::converter::RomanizationEngine;

fn main() {
    let engine = RomanizationEngine::new();
    let test_cases = [
        "a", "aa", "i", "ii", "u", "uu", "e", "ai", "o", "au", "ri", "M", "H", "~",
        "ka", "ki", "ku", "ke", "ko", "kai", "kau", "kra", "malaaii", "aau", "aamaa"
    ];
    for roman in test_cases.iter() {
        let nepali = engine.transliterate_primary(roman);
        println!("{} => {}", roman, nepali);
    }
}
