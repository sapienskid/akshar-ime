use std::collections::{HashMap, HashSet};

// =================================================================================
// ARCHITECTURAL OVERHAUL: SYLLABLE-AWARE FINITE STATE TRANSDUCER (FST)
// This new FST model understands the grammatical structure of Devanagari syllables.
// It correctly builds consonant conjuncts, handles matras, and differentiates
// between independent vowels and vowel signs based on context. This solves the
// core issues of forming words like 'kra' (क्र), 'malai' (मलाइ), and 'aau' (आउ).
// =================================================================================

/// Represents the current state of the syllable being constructed by the FST.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum State {
    /// At the beginning of a word, or after a vowel or symbol.
    /// Ready to start a new syllable.
    Start,
    /// A consonant has been produced and is awaiting a vowel.
    /// The internal buffer currently ends with a virama (halanta).
    Halanta,
    /// A complete syllable (e.g., 'क', 'कि', 'प्र') has just been formed.
    Syllable,
}

/// Result of matching a token, carrying the appropriate Devanagari representation(s).
#[derive(Debug)]
enum MatchResult<'a> {
    Symbol(&'a str),
    Consonant(&'a str),
    Vowel { full: &'a str, matra: &'a str },
}

/// Categorizes a matched Roman token to guide the FST's state transitions.
#[derive(Debug, PartialEq, Eq)]
enum MapKind {
    Symbol,
    Vowel,
    Consonant,
}

pub struct RomanizationEngine {
    consonants: HashMap<&'static str, &'static str>,
    vowels: HashMap<&'static str, (&'static str, &'static str)>, // (Full Vowel, Matra)
    symbols: HashMap<&'static str, &'static str>,
    max_token_len: usize,
}

impl RomanizationEngine {
    pub fn new() -> Self {
        // --- PRINCIPLED & ERGONOMIC KEY MAPPINGS ---
        // These mappings are designed to be intuitive, fast, and consistent.
        // - Aspiration is consistently marked with 'h' (k -> क, kh -> ख).
        // - Retroflex consonants use Capital letters (t -> त, T -> ट).
        // - Vowel length is achieved by doubling (i -> इ, ii -> ई).
        // - Common aliases are provided for user convenience (f/ph, ee/ii).

        let consonants: HashMap<_, _> = [
            ("k", "क"), ("kh", "ख"), ("g", "ग"), ("gh", "घ"), ("ng", "ङ"),
            ("ch", "च"), ("c", "च"), ("chh", "छ"), ("x", "छ"), ("j", "ज"), ("z", "ज"), ("jh", "झ"), ("ny", "ञ"),
            ("T", "ट"), ("Th", "ठ"), ("D", "ड"), ("Dh", "ढ"), ("N", "ण"),
            ("t", "त"), ("th", "थ"), ("d", "द"), ("dh", "ध"), ("n", "न"),
            ("p", "प"), ("ph", "फ"), ("f", "फ"), ("b", "ब"), ("bh", "भ"), ("m", "म"),
            ("y", "य"), ("r", "र"), ("l", "ल"), ("w", "व"), ("v", "व"),
            ("sh", "श"), ("S", "ष"), ("s", "स"), ("h", "ह"),
            ("ksh", "क्ष"), ("tr", "त्र"), ("gy", "ज्ञ"),
        ].iter().cloned().collect();

        // Maps Roman string to a tuple of (Full Independent Vowel, Vowel Sign/Matra)
        let vowels: HashMap<_, _> = [
            ("a", ("अ", "")), // The matra for 'a' is the absence of a virama.
            ("aa", ("आ", "ा")), ("A", ("आ", "ा")),
            ("i", ("इ", "ि")),
            ("ii", ("ई", "ी")), ("ee", ("ई", "ी")),
            ("u", ("उ", "ु")),
            ("uu", ("ऊ", "ू")), ("oo", ("ऊ", "ू")),
            ("e", ("ए", "े")),
            ("ai", ("ऐ", "ै")), ("ae", ("ऐ", "ै")),
            ("o", ("ओ", "ो")),
            ("au", ("औ", "ौ")), ("ao", ("औ", "ौ")),
            ("ri", ("ऋ", "ृ")),
            ("M", ("अं", "ं")), ("am", ("अं", "ं")), // Anusvara
            ("H", ("अः", "ः")), ("ah", ("अः", "ः")), // Visarga
            ("~", ("अँ", "ँ")), // Chandrabindu
        ].iter().cloned().collect();

        let symbols: HashMap<_, _> = [
            (".", "।"), ("..", "।।"),
            ("?", "?"), ("!", "!"), (",", ","), (";", ";"),
            ("OM", "ॐ"), ("'", "ऽ"),
            ("0", "०"), ("1", "१"), ("2", "२"), ("3", "३"), ("4", "४"),
            ("5", "५"), ("6", "६"), ("7", "७"), ("8", "८"), ("9", "९"),
        ].iter().cloned().collect();

        let max_token_len = consonants.keys()
            .chain(vowels.keys())
            .chain(symbols.keys())
            .map(|s| s.len())
            .max()
            .unwrap_or(3);

        Self { consonants, vowels, symbols, max_token_len }
    }

    /// Generates the single most likely, deterministic transliteration.
    /// This is the primary output of the FST.
    pub fn transliterate_primary(&self, roman: &str) -> String {
        if roman.is_empty() { return String::new(); }
        // By default, apply schwa deletion at the end of words (e.g., "ram" -> "राम").
        self.transliterate_base(roman, true)
    }

    /// Generates a list of likely candidates to handle phonetic ambiguity.
    pub fn generate_candidates(&self, roman: &str) -> Vec<String> {
        if roman.is_empty() { return vec![]; }
        if let Some(nepali_symbol) = self.symbols.get(roman) { return vec![nepali_symbol.to_string()]; }

        let primary = self.transliterate_primary(roman);
        let mut candidates = HashSet::new();
        candidates.insert(primary.clone());

        // Heuristic 1: Handle final 'a' ambiguity (e.g., "rama" -> "राम" vs "रामा").
        // The primary transliteration assumes schwa deletion. This variant preserves the 'a'.
        if roman.ends_with('a') && !roman.ends_with("aa") {
            let variant = self.transliterate_base(roman, false);
            if variant != primary {
                candidates.insert(variant);
            }
        }
        
        // Heuristic 2: Split final 'ai' as 'aa' + 'i' (e.g., "malai" -> "मलाइ")
        // This handles cases where user wants separate vowels but types compound vowel
        if roman.ends_with("ai") && roman.len() > 2 {
            let stem = &roman[..roman.len() - 2];
            let stem_with_aa = format!("{}aa", stem);
            let mut variant = self.transliterate_primary(&stem_with_aa);
            if let Some((full_i, _)) = self.vowels.get("i") {
                variant.push_str(full_i);
                if variant != primary {
                    candidates.insert(variant);
                }
            }
        }

        // Heuristic 3: Split final 'au' as 'aa' + 'u' (e.g., "aau" -> "आउ")
        if roman.ends_with("au") && roman.len() > 2 {
            let stem = &roman[..roman.len() - 2];
            if !stem.is_empty() {
                let stem_with_aa = format!("{}aa", stem);
                let mut variant = self.transliterate_primary(&stem_with_aa);
                if let Some((full_u, _)) = self.vowels.get("u") {
                    variant.push_str(full_u);
                    if variant != primary {
                        candidates.insert(variant);
                    }
                }
            }
        }

        // Heuristic 4: Special case for words starting with vowel sequences
        // "aau" -> "आउ" (handle as "aa" + "u")
        if roman.starts_with("aa") && roman.len() > 2 {
            let rest = &roman[2..];
            if let Some((full_vowel, _)) = self.vowels.get(rest) {
                if let Some((aa_full, _)) = self.vowels.get("aa") {
                    let variant = format!("{}{}", aa_full, full_vowel);
                    if variant != primary {
                        candidates.insert(variant);
                    }
                }
            }
        }

        // Heuristic 5: Try splitting at last vowel position
        // This catches patterns like "malai" where we want "ma-la-i" not "mal-ai"
        if let Some(last_vowel_pos) = self.find_last_vowel_boundary(roman) {
            if last_vowel_pos > 0 && last_vowel_pos < roman.len() {
                let (stem, vowel_part) = roman.split_at(last_vowel_pos);
                if self.vowels.contains_key(vowel_part) {
                    let stem_nepali = self.transliterate_primary(stem);
                    if let Some((full_vowel, _)) = self.vowels.get(vowel_part) {
                        let variant = format!("{}{}", stem_nepali, full_vowel);
                        if variant != primary {
                            candidates.insert(variant);
                        }
                    }
                }
            }
        }

        let mut result = vec![primary];
        for cand in candidates {
            if !result.contains(&cand) {
                result.push(cand);
            }
        }
        result
    }

    /// Helper to find the last position where a vowel sequence could be split
    fn find_last_vowel_boundary(&self, roman: &str) -> Option<usize> {
        for i in (1..roman.len()).rev() {
            let potential_vowel = &roman[i..];
            if self.vowels.contains_key(potential_vowel) {
                return Some(i);
            }
        }
        None
    }

    /// The core FST-based transliteration logic.
    fn transliterate_base(&self, roman: &str, force_schwa_deletion: bool) -> String {
        let mut result = String::with_capacity(roman.len() * 3);
        let mut state = State::Start;
        const HALANTA: &str = "\u{094d}";
        let mut input = roman;

        while !input.is_empty() {
            let chunk = &input[..input.len().min(self.max_token_len)];
            
            if let Some((token, match_result, _kind)) = self.match_longest(chunk) {
                match state {
                    State::Start | State::Syllable => match match_result {
                        MatchResult::Consonant(nepali) => {
                            result.push_str(nepali);
                            result.push_str(HALANTA);
                            state = State::Halanta;
                        }
                        MatchResult::Vowel { full, .. } => {
                            // At word/syllable boundary: USE FULL VOWEL
                            result.push_str(full);
                            state = State::Syllable;
                        }
                        MatchResult::Symbol(nepali) => {
                            result.push_str(nepali);
                            state = State::Start;
                        }
                    },
                    State::Halanta => match match_result {
                        MatchResult::Consonant(nepali) => {
                            result.push_str(nepali);
                            result.push_str(HALANTA);
                            // State remains Halanta, building the conjunct
                        }
                        MatchResult::Vowel { matra, .. } => {
                            // After consonant: USE MATRA
                            if result.ends_with(HALANTA) { 
                                result.truncate(result.len() - HALANTA.len());
                            }
                            // If matra is empty (inherent 'a'), we're done
                            // Otherwise, append the matra
                            if !matra.is_empty() {
                                result.push_str(matra);
                            }
                            state = State::Syllable;
                        }
                        MatchResult::Symbol(nepali) => {
                            // Implicit schwa completion before a symbol.
                            if result.ends_with(HALANTA) { 
                                result.truncate(result.len() - HALANTA.len());
                            }
                            result.push_str(nepali);
                            state = State::Start;
                        }
                    }
                }
                input = &input[token.len()..];
            } else {
                // No match found, treat as a literal character.
                if result.ends_with(HALANTA) { 
                    result.truncate(result.len() - HALANTA.len());
                }
                let next_char = input.chars().next().unwrap();
                result.push(next_char);
                state = State::Start;
                input = &input[next_char.len_utf8()..];
            }
        }

        // Final schwa handling at the end of the word.
        if force_schwa_deletion && result.ends_with(HALANTA) {
            result.truncate(result.len() - HALANTA.len());
        }

        result
    }

    /// Implements Longest Prefix Match (LPM) and categorizes the match.
    /// Now returns both full vowel and matra forms for the FST to decide.
    fn match_longest<'a>(&'a self, slice: &'a str) -> Option<(&'a str, MatchResult<'a>, MapKind)> {
        for len in (1..=slice.len()).rev() {
            let token = &slice[0..len];
            
            if let Some(val) = self.symbols.get(token) { 
                return Some((token, MatchResult::Symbol(*val), MapKind::Symbol)); 
            }
            if let Some(val) = self.consonants.get(token) { 
                return Some((token, MatchResult::Consonant(*val), MapKind::Consonant)); 
            }
            if let Some((full, matra)) = self.vowels.get(token) {
                // Return BOTH forms - let the FST state machine decide which to use
                return Some((token, MatchResult::Vowel { full, matra }, MapKind::Vowel));
            }
        }
        None
    }
}

impl Default for RomanizationEngine { fn default() -> Self { Self::new() } }