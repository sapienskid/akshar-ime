// File: src/core/converter.rs
use std::collections::{HashMap, HashSet};

#[derive(Debug, PartialEq, Eq)]
enum State {
    Start,
    Consonant,
}

pub struct RomanizationEngine {
    consonants: HashMap<&'static str, &'static str>,
    full_vowels: HashMap<&'static str, &'static str>,
    vowel_signs: HashMap<&'static str, &'static str>,
    symbols: HashMap<&'static str, &'static str>,
    max_token_len: usize,
}

impl RomanizationEngine {
    pub fn new() -> Self {
        let consonants: HashMap<&'static str, &'static str> = [
            ("k", "क"), ("kh", "ख"), ("g", "ग"), ("gh", "घ"), ("ng", "ङ"),
            ("ch", "च"), ("c", "च"), ("chh", "छ"), ("x", "छ"), ("j", "ज"), ("z", "ज"), ("jh", "झ"),
            ("T", "ट"), ("Th", "ठ"), ("D", "ड"), ("Dh", "ढ"), ("N", "ण"),
            ("t", "त"), ("th", "थ"), ("d", "द"), ("dh", "ध"), ("n", "न"),
            ("p", "प"), ("ph", "फ"), ("f", "फ"), ("b", "ब"), ("bh", "भ"),
            ("m", "म"), ("y", "य"), ("r", "र"), ("l", "ल"), ("w", "व"), ("v", "व"),
            ("s", "स"), ("sh", "श"), ("S", "ष"), ("h", "ह"),
            ("ksh", "क्ष"), ("tr", "त्र"), ("gy", "ज्ञ"),
            ("shree", "श्री"), ("shri", "श्री"),
        ].iter().cloned().collect();
        let full_vowels: HashMap<&'static str, &'static str> = [
            ("a", "अ"), ("aa", "आ"), ("A", "आ"), ("i", "इ"), ("ee", "ई"), ("I", "ई"),
            ("u", "उ"), ("oo", "ऊ"), ("U", "ऊ"), ("e", "ए"), ("ai", "ऐ"),
            ("o", "ओ"), ("au", "औ"), ("ri", "ऋ"), ("R", "ऋ"),
            ("am", "अं"), ("an", "अं"), ("aM", "अं"), ("ah", "अः"), ("aH", "अः"),
        ].iter().cloned().collect();
        let vowel_signs: HashMap<&'static str, &'static str> = [
            ("aa", "ा"), ("A", "ा"), ("i", "ि"), ("ee", "ी"), ("I", "ी"),
            ("u", "ु"), ("oo", "ू"), ("U", "ू"), ("e", "े"), ("ai", "ै"),
            ("o", "ो"), ("au", "ौ"), ("ri", "ृ"), ("R", "ृ"),
            ("M", "ं"), ("H", "ः"), ("~", "ँ"),
        ].iter().cloned().collect();
        let symbols: HashMap<&'static str, &'static str> = [
            (".", "।"), ("|", "।"), ("..", "।।"), ("||", "।।"),
            ("?", "?"), ("!", "!"), (",", ","), (";", ";"),
            ("(", "("), (")", ")"), ("-", "-"), ("_", "_"),
            ("OM", "ॐ"), ("'", "ऽ"),
            ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
            ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९"),
        ].iter().cloned().collect();
        let max_token_len = consonants.keys().chain(full_vowels.keys()).chain(vowel_signs.keys()).chain(symbols.keys()).map(|s| s.len()).max().unwrap_or(1);
        Self { consonants, full_vowels, vowel_signs, symbols, max_token_len }
    }

    pub fn is_symbol(&self, s: &str) -> bool {
        self.symbols.values().any(|&v| v == s)
    }

    pub fn generate_candidates(&self, roman: &str) -> Vec<String> {
        if roman.is_empty() { return vec![]; }
        if let Some(nepali_symbol) = self.symbols.get(roman) {
            return vec![nepali_symbol.to_string()];
        }

        let mut candidate_inputs = HashSet::new();
        candidate_inputs.insert(roman.to_string());

        // Rule 1: 'a' vs 'aa' ambiguity
        if let Some(pos) = roman.rfind('a') {
            if pos == 0 || !roman.get(pos - 1..pos).map_or(false, |c| c == "a") {
                let mut variant = roman.to_string();
                variant.insert(pos + 1, 'a');
                candidate_inputs.insert(variant);
            }
        }
        // Rule 2: 'i' vs 'ee' ambiguity
        if let Some(pos) = roman.rfind('i') {
             if pos == 0 || !roman.get(pos - 1..pos).map_or(false, |c| c == "e") {
                let mut variant = roman.to_string();
                variant.replace_range(pos..pos + 1, "ee");
                candidate_inputs.insert(variant);
            }
        }
        // Rule 3: 'u' vs 'oo' ambiguity
         if let Some(pos) = roman.rfind('u') {
             if pos == 0 || !roman.get(pos - 1..pos).map_or(false, |c| c == "o") {
                let mut variant = roman.to_string();
                variant.replace_range(pos..pos + 1, "oo");
                candidate_inputs.insert(variant);
            }
        }
        // --- NEW: Rule 4: Final consonant 'a' ambiguity (Schwa handling) ---
        // If a word ends in a consonant, it often has an implicit 'a' sound.
        // This rule creates a variant with 'a' appended to handle this.
        // e.g., "sabin" -> "sabina" (for सबिन), "post" -> "posta" (for पोस्ट)
        if let Some(last_char) = roman.chars().last() {
            if last_char.is_ascii_alphabetic() && !"aeiou".contains(last_char) {
                let mut variant = roman.to_string();
                variant.push('a');
                candidate_inputs.insert(variant);
            }
        }

        let mut final_candidates = HashSet::new();
        for input_variant in candidate_inputs {
            final_candidates.insert(self.transliterate_base(&input_variant));
        }
        
        final_candidates.into_iter().collect()
    }

    fn transliterate_base(&self, roman: &str) -> String {
        let mut result = String::new();
        let mut state = State::Start;
        let mut input = roman;
        const HALANTA: char = '\u{094d}';
        while !input.is_empty() {
            let chunk = &input[..input.len().min(self.max_token_len)];
            let mut consumed_len = 1;
            if state == State::Consonant && chunk.starts_with('a') && !self.is_longer_vowel_match(chunk) {
                result.pop();
                state = State::Start;
                consumed_len = 1;
            } else if let Some((token, nepali_str, map_type)) = self.match_longest(chunk) {
                consumed_len = token.len();
                match state {
                    State::Start => {
                        result.push_str(nepali_str);
                        if map_type == "consonant" { result.push(HALANTA); state = State::Consonant; }
                    }
                    State::Consonant => {
                        match map_type {
                            "sign" => { result.pop(); result.push_str(nepali_str); state = State::Start; }
                            "consonant" => { result.push_str(nepali_str); result.push(HALANTA); }
                            _ => { if result.ends_with(HALANTA) { result.pop(); } result.push_str(nepali_str); state = State::Start; }
                        }
                    }
                }
            } else {
                if state == State::Consonant && result.ends_with(HALANTA) { result.pop(); }
                result.push(input.chars().next().unwrap());
                state = State::Start;
            }
            input = &input[consumed_len..];
        }
        if result.ends_with(HALANTA) { result.pop(); }
        result
    }

    fn is_longer_vowel_match(&self, chunk: &str) -> bool {
        if !chunk.starts_with('a') { return false; }
        for len in (2..=chunk.len().min(self.max_token_len)).rev() {
            let token = &chunk[0..len];
            if self.vowel_signs.contains_key(token) || self.full_vowels.contains_key(token) { return true; }
        }
        false
    }

    fn match_longest<'a>(&'a self, slice: &'a str) -> Option<(&'a str, &'a str, &'static str)> {
        for len in (1..=slice.len()).rev() {
            let token = &slice[0..len];
            if let Some(val) = self.symbols.get(token) { return Some((token, val, "symbol")); }
            if let Some(val) = self.vowel_signs.get(token) { return Some((token, val, "sign")); }
            if let Some(val) = self.full_vowels.get(token) { return Some((token, val, "vowel")); }
            if let Some(val) = self.consonants.get(token) { return Some((token, val, "consonant")); }
        }
        None
    }
}

impl Default for RomanizationEngine {
    fn default() -> Self {
        Self::new()
    }
}