use std::collections::{HashMap, HashSet};

// NOTE: The State enum and the struct definition are the same.
// The main changes are in the logic of generating candidates.

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
    // Paste your excellent, comprehensive `new()` function here.
    // It is very good and does not need to be changed.
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
            ("|", "।"), ("||", "।।"), (".", "."), ("..", "।।"), ("?", "?"), ("!", "!"), (",", ","),
            ("OM", "ॐ"), ("'", "ऽ"),
            ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
            ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९"),
        ].iter().cloned().collect();

        let max_token_len = consonants.keys()
            .chain(full_vowels.keys())
            .chain(vowel_signs.keys())
            .chain(symbols.keys())
            .map(|s| s.len())
            .max().unwrap_or(1);

        Self { consonants, full_vowels, vowel_signs, symbols, max_token_len }
    }

    /// **THE NEW ARCHITECTURE**: This function generates variations of the ROMAN input
    /// based on ambiguity rules, transliterates all of them, and returns a unique set.
    pub fn generate_candidates(&self, roman: &str) -> Vec<String> {
        if roman.is_empty() {
            return vec![];
        }

        let mut candidate_inputs = HashSet::new();
        candidate_inputs.insert(roman.to_string());

        // Rule 1: 'a' vs 'aa' ambiguity. If 'a' is present, also try a version with 'aa'.
        // This is crucial for names and places like nepal -> nepaal.
        if let Some(pos) = roman.rfind('a') {
            if pos == 0 || !roman.get(pos-1..pos).map_or(false, |c| c == "a") {
                let mut variant = roman.to_string();
                variant.insert(pos + 1, 'a'); // nepal -> nepaal
                candidate_inputs.insert(variant);
            }
        }
        
        // Rule 2: 'i' vs 'ee' ambiguity
        if let Some(pos) = roman.rfind('i') {
             if pos == 0 || !roman.get(pos-1..pos).map_or(false, |c| c == "e") {
                let mut variant = roman.to_string();
                variant.replace_range(pos..pos+1, "ee"); // sabin -> sabeen
                candidate_inputs.insert(variant);
            }
        }

        // Rule 3: 'u' vs 'oo' ambiguity
         if let Some(pos) = roman.rfind('u') {
             if pos == 0 || !roman.get(pos-1..pos).map_or(false, |c| c == "o") {
                let mut variant = roman.to_string();
                variant.replace_range(pos..pos+1, "oo"); // pukar -> pookar
                candidate_inputs.insert(variant);
            }
        }

        // Transliterate all generated Roman variations and collect unique Nepali results
        let mut final_candidates = HashSet::new();
        for input_variant in candidate_inputs {
            final_candidates.insert(self.transliterate_base(&input_variant));
        }
        
        final_candidates.into_iter().collect()
    }

    /// The core FSM transliterator. Its job is to be as literal as possible.
    fn transliterate_base(&self, roman: &str) -> String {
        // This function is now the "worker" and doesn't need to change.
        // You can paste your last working version of transliterate_base here.
        // For clarity, I am providing the correct one again.
        let mut result = String::new();
        let mut state = State::Start;
        let mut input = roman;
        const HALANTA: char = '\u{094d}';

        while !input.is_empty() {
            let chunk = &input[..input.len().min(self.max_token_len)];
            let mut consumed_len = 1;

            if state == State::Consonant && chunk.starts_with('a') && !self.is_longer_vowel_match(chunk) {
                result.pop(); // Remove halanta
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

    // Your helper methods are perfect, keep them.
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