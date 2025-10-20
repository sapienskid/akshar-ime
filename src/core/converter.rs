use std::collections::{HashMap, HashSet};

/// A robust three-state automaton for context-aware Indic script transliteration.
/// This state machine correctly differentiates between vowels at the start of a word,
/// vowel signs (matras) following a consonant, and full vowels following a completed syllable.
/// This design is a direct implementation of the Finite-State Transducer principles.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum State {
    Start,
    Consonant,
    Vowel,
}

/// Categorizes the type of mapping to guide the FST's state transitions.
#[derive(Debug, PartialEq, Eq)]
enum MapKind {
    Symbol,
    FullVowel,
    VowelSign,
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
        // =================================================================================
        // ARCHITECTURAL OVERHAUL: VOWEL MAPPING BASED ON DETAILED RESEARCH REPORT
        // The FST logic is sound, but its accuracy depends entirely on the completeness
        // of these rule tables. The following maps are now comprehensive.
        // =================================================================================

        let consonants: HashMap<_, _> = [
            ("k", "क"), ("K", "क"), ("kh", "ख"), ("KH", "ख"), ("Kh", "ख"), ("g", "ग"), ("G", "ग"), ("gh", "घ"), ("GH", "घ"), ("Gh", "घ"), ("ng", "ङ"), ("NG", "ङ"),
            ("ch", "च"), ("CH", "च"), ("Ch", "च"), ("c", "च"), ("C", "च"), ("chh", "छ"), ("CHH", "छ"), ("Chh", "छ"), ("x", "छ"), ("X", "छ"), ("j", "ज"), ("J", "ज"), ("z", "ज"), ("Z", "ज"), ("jh", "झ"), ("JH", "झ"), ("Jh", "झ"),
            ("T", "ट"), ("Th", "ठ"), ("TH", "ठ"), ("D", "ड"), ("Dh", "ढ"), ("DH", "ढ"), ("N", "ण"),
            ("t", "त"), ("th", "थ"), ("d", "द"), ("dh", "ध"), ("n", "न"),
            ("p", "प"), ("P", "प"), ("ph", "फ"), ("f", "फ"), ("F", "फ"), ("b", "ब"), ("B", "ब"), ("bh", "भ"), ("BH", "भ"), ("Bh", "भ"),
            ("m", "म"), ("M", "म"), ("y", "य"), ("Y", "य"), ("r", "र"), ("R", "र"), ("l", "ल"), ("L", "ल"), ("w", "व"), ("W", "व"), ("v", "व"), ("V", "व"),
            ("s", "स"), ("sh", "श"), ("SH", "श"), ("Sh", "श"), ("S", "ष"), ("h", "ह"), ("H", "ह"),
            ("ksh", "क्ष"), ("KSH", "क्ष"), ("Ksh", "क्ष"), ("tr", "त्र"), ("TR", "त्र"), ("Tr", "त्र"), ("gy", "ज्ञ"), ("GY", "ज्ञ"), ("Gy", "ज्ञ"),
        ].iter().cloned().collect();

        // Full (Independent) Vowels: Used when FST state is `Start` or `Vowel`.
        let full_vowels: HashMap<_, _> = [
            ("a", "अ"),
            ("aa", "आ"), ("AA", "आ"),
            ("i", "इ"),
            ("ee", "ई"), ("EE", "ई"), ("ii", "ई"), ("II", "ई"), // Added 'ii' alias
            ("u", "उ"),
            ("oo", "ऊ"), ("OO", "ऊ"), ("uu", "ऊ"), ("UU", "ऊ"), // Added 'uu' alias
            ("e", "ए"), // NOTE: Per convention. User spec had `a` -> `ए` which would conflict.
            ("ai", "ऐ"), ("AI", "ऐ"), ("ae", "ऐ"), ("AE", "ऐ"), // Added 'ae' alias
            ("o", "ओ"),
            ("au", "औ"), ("AU", "औ"), ("ao", "औ"), ("AO", "औ"), // Added 'ao' alias
            ("am", "अं"), ("AM", "अं"), ("aM", "अं"), ("an", "अं"), ("AN", "अं"),
            ("ah", "अः"), ("AH", "अः"), ("a:", "अः"), // Added 'a:' alias
            ("ri", "ऋ"), ("RI", "ऋ"),
            ("rr", "ॠ"), ("RR", "ॠ"), // Added long vocalic R
        ].iter().cloned().collect();

        // Dependent Vowels (Matras): Used ONLY when FST state is `Consonant`.
        let vowel_signs: HashMap<_, _> = [
            ("a", ""), // 'a' removes the halanta (inherent schwa)
            ("aa", "ा"), ("AA", "ा"),
            ("i", "ि"),
            ("ee", "ी"), ("EE", "ी"), ("ii", "ी"), ("II", "ी"), // Added 'ii' alias
            ("u", "ु"),
            ("oo", "ू"), ("OO", "ू"), ("uu", "ू"), ("UU", "ू"), // Added 'uu' alias
            ("e", "े"), // Standard 'e' matra
            ("E", "ॅ"), // For loanwords like 'cat' -> 'kEṭ' -> 'कैट'
            ("ai", "ै"), ("AI", "ै"), ("ae", "ै"), ("AE", "ै"), // Added 'ae' alias
            ("o", "ो"), // Standard 'o' matra
            ("O", "ॉ"), // For loanwords like 'cot' -> 'kOṭ' -> 'कॉट'
            ("au", "ौ"), ("AU", "ौ"), ("ao", "ौ"), ("AO", "ौ"), // Added 'ao' alias
            ("r", "ृ"), ("ri", "ृ"), ("RI", "ृ"), ("R", "ृ"),
            ("rr", "ॄ"), ("RR", "ॄ"), // Added long vocalic R matra
            ("M", "ं"), // Anusvara
            ("H", "ः"), // Visarga
            ("~", "ँ"), // Chandrabindu
        ].iter().cloned().collect();

        let symbols: HashMap<_, _> = [
            (".", "।"), ("|", "।"), ("..", "।।"), ("||", "।।"),
            ("?", "?"), ("!", "!"), (",", ","), (";", ";"),
            ("OM", "ॐ"), ("'", "ऽ"),
            ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
            ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९"),
        ].iter().cloned().collect();

        // Dynamically calculate max_token_len to ensure Longest Prefix Match is always correct.
        let max_token_len = consonants.keys()
            .chain(full_vowels.keys())
            .chain(symbols.keys())
            .map(|s| s.len())
            .max()
            .unwrap_or(3);

        Self { consonants, full_vowels, vowel_signs, symbols, max_token_len }
    }

    /// Generates the single most likely, deterministic transliteration.
    pub fn transliterate_primary(&self, roman: &str) -> String {
        if roman.is_empty() { return String::new(); }
        let force = roman.ends_with('a') && !roman.ends_with("aa");
        self.transliterate_base(roman, force)
    }

    /// Generates a list of likely candidates, including the primary one and common variants.
    pub fn generate_candidates(&self, roman: &str) -> Vec<String> {
        if roman.is_empty() { return vec![]; }
        if let Some(nepali_symbol) = self.symbols.get(roman) { return vec![nepali_symbol.to_string()]; }

        let primary = self.transliterate_primary(roman);
        let mut candidates = HashSet::new();
        candidates.insert(primary.clone());

        // Heuristic 1: Handle final 'a' ambiguity (e.g., "rama" -> "राम" vs "रामा")
        if roman.ends_with('a') && !roman.ends_with("aa") {
            let variant = self.transliterate_base(roman, false);
            if variant != primary {
                candidates.insert(variant);
            }
        }
        
        // Heuristic 2: Handle schwa-vowel boundary ambiguity (e.g., "malai" -> "मलै" vs "मलाइ")
        if let Some(last_vowel_pos) = roman.rfind(|c: char| "aeiouAEIOU".contains(c)) {
            if last_vowel_pos > 0 {
                 let (stem, last_vowel) = roman.split_at(last_vowel_pos);
                 // Ensure the 'last_vowel' part isn't a consonant itself (like 'ch' or 'sh')
                 if !self.consonants.contains_key(last_vowel) {
                    let stem_nepali = self.transliterate_primary(stem);
                    let vowel_nepali = self.transliterate_primary(last_vowel);
                    let variant = format!("{}{}", stem_nepali, vowel_nepali);
                    if variant != primary {
                        candidates.insert(variant);
                    }
                 }
            }
        }

        // Collect into a vec, ensuring the primary transliteration is first.
        let mut result = vec![primary];
        for cand in candidates {
            if !result.contains(&cand) {
                result.push(cand);
            }
        }
        result
    }

    /// The core FST-based transliteration logic. (LOGIC UNCHANGED, DATA IS NEW)
    fn transliterate_base(&self, roman: &str, force_final_a_matra: bool) -> String {
        let roman = roman.to_lowercase();
        let mut result = String::with_capacity(roman.len() * 3);
        let mut state = State::Start;
        const HALANTA: &str = "\u{094d}";
        let mut input = roman.as_str();

        while !input.is_empty() {
            let remaining_len = input.len();
            let slice_len = remaining_len.min(self.max_token_len);
            let chunk = &input[..slice_len];

            let force_aa = force_final_a_matra && remaining_len == 1 && chunk.starts_with('a');
            let effective_chunk = if force_aa { "aa" } else { chunk };

            if let Some((token, nepali_str, kind)) = self.match_longest(effective_chunk, state) {
                let consumed_len = if force_aa { 1 } else { token.len() };

                match kind {
                    MapKind::Symbol => {
                        if state == State::Consonant && result.ends_with(HALANTA) {
                            result.pop();
                        }
                        result.push_str(nepali_str);
                        state = State::Start;
                    }
                    MapKind::FullVowel => {
                        if state == State::Consonant && result.ends_with(HALANTA) {
                            result.pop();
                        }
                        result.push_str(nepali_str);
                        state = State::Vowel;
                    }
                    MapKind::VowelSign => {
                        if state == State::Consonant && result.ends_with(HALANTA) {
                            result.pop();
                            result.push_str(nepali_str);
                            state = State::Vowel;
                        } else {
                            if let Some(fv) = self.full_vowels.get(token) {
                                result.push_str(fv);
                                state = State::Vowel;
                            }
                        }
                    }
                    MapKind::Consonant => {
                        result.push_str(nepali_str);
                        result.push_str(HALANTA);
                        state = State::Consonant;
                    }
                }
                input = &input[consumed_len..];
            } else {
                if state == State::Consonant && result.ends_with(HALANTA) {
                    result.pop();
                }
                let next_char = input.chars().next().unwrap();
                result.push(next_char);
                state = State::Start;
                input = &input[next_char.len_utf8()..];
            }
        }

        if result.ends_with(HALANTA) {
            result.pop();
        }

        result
    }

    /// Implements Longest Prefix Match (LPM) based on the FST's current state. (UNCHANGED)
    fn match_longest<'a>(&'a self, slice: &'a str, state: State) -> Option<(&'a str, &'a str, MapKind)> {
        for len in (1..=slice.len()).rev() {
            let token = &slice[0..len];

            if let Some(val) = self.symbols.get(token) { return Some((token, *val, MapKind::Symbol)); }

            match state {
                State::Start | State::Vowel => {
                    if let Some(val) = self.full_vowels.get(token) { return Some((token, *val, MapKind::FullVowel)); }
                    if let Some(val) = self.consonants.get(token) { return Some((token, *val, MapKind::Consonant)); }
                }
                State::Consonant => {
                    if let Some(val) = self.vowel_signs.get(token) { return Some((token, *val, MapKind::VowelSign)); }
                    if let Some(val) = self.consonants.get(token) { return Some((token, *val, MapKind::Consonant)); }
                }
            }
        }
        None
    }
}

impl Default for RomanizationEngine { fn default() -> Self { Self::new() } }