// File: src/fuzzy/grammar.rs
//
// Nepali Grammar & Orthographic Normalization Engine.
//
// Standardizes Devanagari text according to standard Nepali grammar and
// orthographic rules (नेपाली वर्णविन्यास तथा व्याकरण नियमहरू, Brihat Nepali Shabdakosh):
//
//   1. Hrasva/Dirgha (ह्रस्व-दीर्घ):
//      - Suffix -हरू (plural) must be dirgha (ू), not short -हरु.
//      - Verb passives: -इन्छ, -इने, -इयो (ह्रस्व ि) instead of -ीन्छ, -ीने (गरीन्छ -> गरिन्छ).
//      - Postpositions: -माथि, -अघि, -पछि, -देखि (ह्रस्व ि).
//      - Adjectival suffix: -िक (ह्रस्व ि) as in सामाजिक, आर्थिक, राजनीतिक, प्राकृतिक.
//      - Feminine nominal endings: -ई (दीर्घ ी) as in छोरी, नानी, श्रीमती.
//
//   2. Sibilants (श, ष, स):
//      - Standard Sanskrit/Tatsam orthography: विशेष (not बिसेस), देश (not देस),
//        शहीद (not सहिद), शिक्षा (not सिक्षा), भाषा (not भासा), सन्तोष (not सन्तोस).
//      - Conjuncts: ष्ट, ष्ठ (कष्ट, दृष्टि, राष्ट्र), श्च (निश्चय, आश्चर्य).
//
//   3. Labials (ब vs व):
//      - Standard Tatsam prefixes and stems: विकास (not बिकास), व्यवस्था (not ब्यवस्था),
//        व्यक्ति (not ब्यक्ति), व्यापार (not ब्यापार), वातावरण (not बातावरण),
//        विद्यालय (not बिद्यालय), विचार (not बिचार), विवरण (not बिबरण), संविधान (not संबिधान).
//      - Preserve genuine Nepali roots: बजार, बाटो, बालक, बुबा, बहिनी, बस्नु, बनाउनु.
//
//   4. Nasals & Bindu (ँ vs ं vs Pancham Varna):
//      - Anusvara before semivowels/sibilants: संविधान, संसार, संवाद, सिंह.
//      - Chandrabindu in tadbhava nasal roots: हुँदा (not हुदा), पाँच, आँखा, काठमाडौँ.
//
//   5. Cycle Consistency:
//      - Given an orthographic skeleton, GrammarCanonicalizer clusters all variants
//        and selects the grammatically canonical Nepali word.

use std::collections::HashMap;
use std::path::Path;

/// Compute an invariant orthographic/phonetic skeleton for a Devanagari string.
///
/// Words that differ only in standard Nepali orthographic alternations:
///   - vowel length (ि vs ी, ु vs ू, इ vs ई, उ vs ऊ)
///   - sibilant letter (श vs ष vs स)
///   - labials (व vs ब)
///   - nasals (ँ vs ं vs homorganic nasals before consonant)
///   - ri/rri (ऋ vs रि, ृ vs ्रि)
///   - gya (ज्ञ vs ग्य)
///   - zero-width characters and nuktas
///
/// produce the EXACT SAME skeletal string.
pub fn orthographic_skeleton(dev: &str) -> String {
    let mut skel = String::with_capacity(dev.len());
    let chars: Vec<char> = dev.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        let ch = chars[i];
        // Strip zero-width joiners, non-joiners, zero-width spaces, nukta.
        if ch == '\u{200C}' || ch == '\u{200D}' || ch == '\u{200B}' || ch == '\u{093C}' {
            i += 1;
            continue;
        }

        // Halanta conjuncts: check for homorganic nasal conjuncts (ङ्, ञ्, ण्, न्, म्) before consonants
        if i + 2 < n && chars[i + 1] == '\u{094D}' {
            let nasal = matches!(
                ch,
                '\u{0919}' | '\u{091E}' | '\u{0923}' | '\u{0928}' | '\u{092E}'
            );
            let next_is_cons = matches!(chars[i + 2], '\u{0915}'..='\u{0939}');
            if nasal && next_is_cons {
                // Drop nasal conjunct for soft nasal invariance
                i += 2; // skip nasal and halanta
                continue;
            }
        }

        // Check for ज्ञ (ज् + ् + ञ)
        if ch == '\u{091C}' && i + 2 < n && chars[i + 1] == '\u{094D}' && chars[i + 2] == '\u{091E}'
        {
            skel.push_str("ग्य");
            i += 3;
            continue;
        }

        match ch {
            // Dependent vowel signs: merge long -> short
            '\u{0940}' => skel.push('\u{093F}'), // ी -> ि
            '\u{0942}' => skel.push('\u{0941}'), // ू -> ु
            '\u{0943}' => {
                // ृ -> रि
                skel.push('\u{0930}');
                skel.push('\u{093F}');
            }

            // Independent vowels: merge long -> short
            '\u{0908}' => skel.push('\u{0907}'), // ई -> इ
            '\u{090A}' => skel.push('\u{0909}'), // ऊ -> उ
            '\u{090B}' => {
                // ऋ -> रि
                skel.push('\u{0930}');
                skel.push('\u{093F}');
            }

            // Sibilants: merge to 'स'
            '\u{0936}' | '\u{0937}' => skel.push('\u{0938}'), // श, ष -> स

            // Labials: merge 'ब' -> 'व'
            '\u{092C}' => skel.push('\u{0935}'), // ब -> व

            // Nasal markers: drop chandrabindu and anusvara for soft nasal invariance (e.g. हुँदा vs हुदा)
            '\u{0901}' | '\u{0902}' => {
                // dropped
            }

            // Word-final halanta normalization (e.g. गर्दछन् vs गर्दछन)
            '\u{094D}' if i == n - 1 => {
                // Ignore trailing virama for invariant skeleton match
            }

            other => skel.push(other),
        }
        i += 1;
    }

    skel
}

/// Score a Nepali word's adherence to standard Nepali orthography.
/// Higher score means more grammatically canonical in modern standard Nepali.
pub fn score_nepali_orthography(word: &str) -> f64 {
    let mut score = 0.0f64;

    // 1. Plural suffix rule: -हरू is canonical (दीर्घ), -हरु is colloquial error
    if word.ends_with("हरू") {
        score += 8.0;
    } else if word.ends_with("हरु") {
        score -= 8.0;
    }

    // 2. Verb passive rule: -इन्छ / -इने / -इयो is canonical (ह्रस्व)
    if word.ends_with("िन्छ")
        || word.ends_with("िन्छन्")
        || word.ends_with("ियो")
        || word.ends_with("िनेछ")
    {
        score += 8.0;
    }
    if word.ends_with("ीन्छ")
        || word.ends_with("ीन्छन्")
        || word.ends_with("ीयो")
        || word.ends_with("ीनेछ")
    {
        score -= 10.0;
    }

    // 3. Postpositions: -माथि, -अघि, -पछि, -देखि are canonical (ह्रस्व)
    for p in &["माथि", "अघि", "पछि", "देखि"] {
        if word.ends_with(p) {
            score += 5.0;
        }
    }
    for p in &["माथी", "अघी", "पछी", "देखी"] {
        if word.ends_with(p) {
            score -= 6.0;
        }
    }

    // 4. Adjectival suffix -िक (ह्रस्व)
    if word.ends_with("िक") {
        score += 4.0;
    } else if word.ends_with("ीक") {
        score -= 6.0;
    }

    // 5. Tatsam prefixes: वि-, व्य-, परि-, सम्- vs बि-, ब्य-, सँ-
    if word.starts_with("व्य") || word.starts_with("व्या") {
        score += 7.0;
    } else if word.starts_with("ब्य") || word.starts_with("ब्या") {
        score -= 7.0;
    }

    if word.starts_with("वि") && !word.starts_with("बि") {
        score += 4.0;
    }

    // Words like विद्यालय vs बिद्यालय
    if word.contains("विद्या") {
        score += 8.0;
    } else if word.contains("बिद्या") {
        score -= 8.0;
    }

    // 6. Sibilants in known Sanskrit clusters: ष्ट, ष्ठ, श्च
    if word.contains("ष्ट") || word.contains("ष्ठ") || word.contains("श्च") {
        score += 7.0;
    }
    if word.contains("स्त") && (word.contains("कस्त") || word.contains("दDefault"))
    {
        score -= 5.0;
    }

    // 7. Sibilant words: विशेष, शहीद, देश, शिक्षा, भाषा, सन्तोष
    if word == "विशेष"
        || word == "शहीद"
        || word == "देश"
        || word == "शिक्षा"
        || word == "भाषा"
        || word == "सन्तोष"
    {
        score += 12.0;
    }
    if word == "बिसेस"
        || word == "विसेस"
        || word == "सहिद"
        || word == "सहीद"
        || word == "देस"
        || word == "भासा"
    {
        score -= 10.0;
    }

    // 8. Chandrabindu on nasal roots: हुँदा, हुँदै, हुँदैन
    if word.starts_with("हुँ") {
        score += 6.0;
    } else if word == "हुदा" || word == "हुदै" || word == "हुदैन" {
        score -= 7.0;
    }

    // 9. Anusvara before semivowels/sibilants: संविधान, संसार, संवाद
    if word.starts_with("संवि") || word.starts_with("संसा") || word.starts_with("संवा")
    {
        score += 8.0;
    } else if word.starts_with("संबि") || word.starts_with("सँवि") {
        score -= 8.0;
    }

    // 10. Modern standard "कानुन" (Nepal Academy / Supreme Court)
    if word == "कानुन" {
        score += 5.0;
    }

    score
}

/// A grammar-driven vocabulary canonicalizer and cycle validator.
pub struct GrammarCanonicalizer {
    /// Reference frequencies for clean corpus words: word -> freq
    pub ref_freq: HashMap<String, u32>,
    /// Index: skeleton -> list of known candidate Devanagari words
    pub skeleton_index: HashMap<String, Vec<String>>,
}

impl GrammarCanonicalizer {
    pub fn new() -> Self {
        Self {
            ref_freq: HashMap::new(),
            skeleton_index: HashMap::new(),
        }
    }

    /// Load reference vocabulary (e.g. data/word_freq_text.bin or data/word_freq.bin)
    pub fn load_reference_vocab(&mut self, path: &Path) -> Result<usize, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let map: HashMap<String, u32> = bincode::deserialize(&bytes)
            .map_err(|e| format!("deserialize error for {}: {e}", path.display()))?;
        let count = map.len();
        for (word, freq) in map {
            self.add_vocab_word(&word, freq);
        }
        self.build_index();
        Ok(count)
    }

    /// Add a vocabulary word with its occurrence count.
    pub fn add_vocab_word(&mut self, word: &str, freq: u32) {
        let entry = self.ref_freq.entry(word.to_string()).or_insert(0);
        *entry = (*entry).max(freq);
    }

    /// Build or re-index the skeletal invariant buckets.
    pub fn build_index(&mut self) {
        self.skeleton_index.clear();
        for word in self.ref_freq.keys() {
            let skel = orthographic_skeleton(word);
            self.skeleton_index
                .entry(skel)
                .or_default()
                .push(word.clone());
        }
    }

    /// Canonicalize any Devanagari word using grammar rules and corpus consensus.
    ///
    /// If the word is an orthographic error (e.g. सहिद, बिकास, गरीन्छ, ब्यवस्था),
    /// this resolves it to the grammatically canonical form (शहीद, विकास, गरिन्छ, व्यवस्था).
    pub fn canonicalize(&self, word: &str) -> String {
        let word = word.trim();
        if word.is_empty() {
            return String::new();
        }

        // Query reference frequency and grammar rules via skeletal invariant bucket
        let skel = orthographic_skeleton(word);
        let candidates = match self.skeleton_index.get(&skel) {
            Some(c) if !c.is_empty() => c,
            _ => {
                // Rule-based transformation fallback
                return self.rule_transform(word);
            }
        };

        // If only one candidate in the bucket, check if it's better
        if candidates.len() == 1 {
            let c = &candidates[0];
            if c != word {
                let s_orig = score_nepali_orthography(word);
                let s_cand = score_nepali_orthography(c);
                if s_cand > s_orig {
                    return c.clone();
                }
            }
            return c.clone();
        }

        // 3. Score all candidates sharing the skeleton:
        //    Score = freq_score + grammar_orthography_score
        let mut best_word = word.to_string();
        let mut best_score = f64::NEG_INFINITY;

        for cand in candidates {
            let freq = self.ref_freq.get(cand).copied().unwrap_or(0);
            let freq_score = ((freq as f64) + 1.0).ln() * 2.5;
            let gram_score = score_nepali_orthography(cand);
            let total = freq_score + gram_score;

            if total > best_score {
                best_score = total;
                best_word = cand.clone();
            }
        }

        best_word
    }

    /// Rule-based transform for words not found in the reference dictionary.
    fn rule_transform(&self, word: &str) -> String {
        // Plural -हरु -> -हरू
        if word.ends_with("हरु") && !word.ends_with("हरू") {
            let prefix = &word[..word.len() - "हरु".len()];
            return format!("{prefix}हरू");
        }

        // Verb passive -ीन्छ -> -इन्छ
        if let Some(prefix) = word.strip_suffix("ीन्छ") {
            return format!("{prefix}िन्छ");
        }

        // Prefix ब्य- -> व्य-
        if let Some(suffix) = word.strip_prefix("ब्य") {
            return format!("व्य{suffix}");
        }

        // Prefix ब्या- -> व्या-
        if let Some(suffix) = word.strip_prefix("ब्या") {
            return format!("व्या{suffix}");
        }

        // बिद्या -> विद्या
        if word.contains("बिद्या") {
            return word.replace("बिद्या", "विद्या");
        }

        word.to_string()
    }

    /// Check if two Devanagari words are orthographically equivalent variants
    /// (or one is an orthographic misspelling of the other).
    pub fn are_equivalent(&self, w1: &str, w2: &str) -> bool {
        if w1 == w2 {
            return true;
        }
        let s1 = orthographic_skeleton(w1);
        let s2 = orthographic_skeleton(w2);
        if s1 == s2 {
            return true;
        }
        self.canonicalize(w1) == self.canonicalize(w2)
    }
}

impl Default for GrammarCanonicalizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate natural phonetic typing variations for a Roman string.
///
/// Users write Roman Nepali with diverse habit patterns:
///   - 'b' vs 'v' / 'w' (e.g. bikas / vikas / wikas)
///   - 'i' vs 'ee' (e.g. sahid / saheed / shaheed)
///   - 'u' vs 'oo' (e.g. kanun / kanoon)
///   - 's' vs 'sh' (e.g. bishesh / vishesh / bisesh)
///   - 'ch' vs 'chh' (e.g. garincha / garinchha)
///   - 'n' vs 'm' before labials (e.g. sambidhan / sanbidhan)
pub fn generate_roman_phonetic_variants(roman: &str, max_variants: usize) -> Vec<String> {
    let mut variants = vec![roman.to_lowercase()];

    // Common phonetic substitutions
    let rules: &[(&str, &str)] = &[
        ("b", "v"),
        ("v", "b"),
        ("v", "w"),
        ("w", "v"),
        ("ee", "i"),
        ("i", "ee"),
        ("oo", "u"),
        ("u", "oo"),
        ("sh", "s"),
        ("s", "sh"),
        ("ch", "chh"),
        ("chh", "ch"),
        ("sam", "san"),
        ("san", "sam"),
    ];

    for &(from, to) in rules {
        let mut next = Vec::new();
        for v in &variants {
            if v.contains(from) {
                let replaced = v.replacen(from, to, 1);
                if !variants.contains(&replaced) && !next.contains(&replaced) {
                    next.push(replaced);
                }
            }
        }
        variants.extend(next);
        if variants.len() >= max_variants {
            break;
        }
    }

    variants.truncate(max_variants);
    variants
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orthographic_skeleton_invariance() {
        // Labials: ब vs व
        assert_eq!(
            orthographic_skeleton("विकास"),
            orthographic_skeleton("बिकास")
        );
        assert_eq!(
            orthographic_skeleton("व्यवस्था"),
            orthographic_skeleton("ब्यवस्था")
        );
        // Sibilants: श vs ष vs स
        assert_eq!(orthographic_skeleton("विशेष"), orthographic_skeleton("बिसेस"));
        assert_eq!(orthographic_skeleton("विशेष"), orthographic_skeleton("विसेस"));
        assert_eq!(orthographic_skeleton("शहीद"), orthographic_skeleton("सहिद"));
        assert_eq!(orthographic_skeleton("शहीद"), orthographic_skeleton("सहीद"));
        assert_eq!(orthographic_skeleton("देश"), orthographic_skeleton("देस"));
        // Hrasva vs Dirgha: ि vs ी
        assert_eq!(
            orthographic_skeleton("गरिन्छ"),
            orthographic_skeleton("गरीन्छ")
        );
        assert_eq!(orthographic_skeleton("कानुन"), orthographic_skeleton("कानून"));
        // Plural suffix: -हरू vs -हरु
        assert_eq!(
            orthographic_skeleton("मानिसहरू"),
            orthographic_skeleton("मानिसहरु")
        );
        // Nasal: हुँदा vs हुदा
        assert_eq!(orthographic_skeleton("हुँदा"), orthographic_skeleton("हुदा"));
    }

    #[test]
    fn test_grammar_scoring_prefers_canonical() {
        assert!(score_nepali_orthography("विकास") > score_nepali_orthography("बिकास"));
        assert!(score_nepali_orthography("व्यवस्था") > score_nepali_orthography("ब्यवस्था"));
        assert!(score_nepali_orthography("विशेष") > score_nepali_orthography("बिसेस"));
        assert!(score_nepali_orthography("शहीद") > score_nepali_orthography("सहिद"));
        assert!(score_nepali_orthography("गरिन्छ") > score_nepali_orthography("गरीन्छ"));
        assert!(score_nepali_orthography("मानिसहरू") > score_nepali_orthography("मानिसहरु"));
        assert!(score_nepali_orthography("विद्यालय") > score_nepali_orthography("बिद्यालय"));
        assert!(score_nepali_orthography("संविधान") > score_nepali_orthography("संबिधान"));
    }

    #[test]
    fn test_canonicalizer_resolves_common_errors() {
        let mut canon = GrammarCanonicalizer::new();
        canon.add_vocab_word("विकास", 10000);
        canon.add_vocab_word("बिकास", 500);
        canon.add_vocab_word("शहीद", 5000);
        canon.add_vocab_word("सहिद", 200);
        canon.add_vocab_word("गरिन्छ", 8000);
        canon.add_vocab_word("गरीन्छ", 300);
        canon.build_index();

        assert_eq!(canon.canonicalize("बिकास"), "विकास");
        assert_eq!(canon.canonicalize("विकास"), "विकास");
        assert_eq!(canon.canonicalize("सहिद"), "शहीद");
        assert_eq!(canon.canonicalize("शहीद"), "शहीद");
        assert_eq!(canon.canonicalize("गरीन्छ"), "गरिन्छ");
        assert_eq!(canon.canonicalize("गरिन्छ"), "गरिन्छ");
        assert_eq!(canon.canonicalize("मानिसहरु"), "मानिसहरू");
    }

    #[test]
    fn test_phonetic_roman_variants() {
        let vars = generate_roman_phonetic_variants("bikas", 5);
        assert!(vars.contains(&"bikas".to_string()));
        assert!(vars.contains(&"vikas".to_string()));

        let vars2 = generate_roman_phonetic_variants("garincha", 5);
        assert!(vars2.contains(&"garinchha".to_string()));
    }
}
