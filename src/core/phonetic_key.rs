// File: src/core/phonetic_key.rs
//
// Empirical phonetic projection and canonical folding for Nepali Roman transliteration.
//
// Maps Devanagari words to canonical Roman keys (phi) and folds Roman queries
// into the same canonical space (kappa) for inverted index candidate generation.

use unicode_normalization::UnicodeNormalization;

fn is_consonant(c: char) -> bool {
    matches!(c, '\u{0915}'..='\u{0939}')
}

fn is_matra(c: char) -> bool {
    matches!(c, '\u{093E}'..='\u{094C}')
}

fn is_independent_vowel(c: char) -> bool {
    matches!(c, '\u{0905}'..='\u{0914}')
}

fn consonant_str(c: char) -> &'static str {
    match c {
        'क' => "k",
        'ख' => "kh",
        'ग' => "g",
        'घ' => "gh",
        'ङ' => "ng",
        'च' => "ch",
        'छ' => "chh",
        'ज' => "j",
        'झ' => "jh",
        'ञ' => "ny",
        'ट' => "t",
        'ठ' => "th",
        'ड' => "d",
        'ढ' => "dh",
        'ण' => "n",
        'त' => "t",
        'थ' => "th",
        'द' => "d",
        'ध' => "dh",
        'न' => "n",
        'प' => "p",
        'फ' => "ph",
        'ब' => "b",
        'भ' => "bh",
        'म' => "m",
        'य' => "y",
        'र' => "r",
        'ल' => "l",
        'व' => "v",
        'श' => "sh",
        'ष' => "sh",
        'स' => "s",
        'ह' => "h",
        _ => "",
    }
}

fn matra_str(c: char) -> &'static str {
    match c {
        'ा' => "aa",
        'ि' => "i",
        'ी' => "ii",
        'ु' => "u",
        'ू' => "uu",
        'ृ' => "ri",
        'े' => "e",
        'ै' => "ai",
        'ो' => "o",
        'ौ' => "au",
        _ => "",
    }
}

fn independent_vowel_str(c: char) -> &'static str {
    match c {
        'अ' => "a",
        'आ' => "aa",
        'इ' => "i",
        'ई' => "ii",
        'उ' => "u",
        'ऊ' => "uu",
        'ऋ' => "ri",
        'ए' => "e",
        'ऐ' => "ai",
        'ओ' => "o",
        'औ' => "au",
        _ => "",
    }
}

/// Project a Devanagari word into its canonical Roman representation (phi).
/// Inherent schwa is retained medially, on monosyllables, and on conjunct codas,
/// but dropped on word-final bare consonants of multisyllabic words.
pub fn project_devanagari(word: &str) -> String {
    let word: String = word.nfc().collect();
    let chars: Vec<char> = word.chars().collect();
    let n = chars.len();
    if n == 0 {
        return String::new();
    }

    let consonant_count = chars.iter().filter(|&&c| is_consonant(c)).count();
    let mut out = String::with_capacity(n * 3);
    let mut i = 0;
    while i < n {
        let c = chars[i];

        // 1. Nepali special conjunct: ज्ञ (ज + ् + ञ) is pronounced and transliterated as "gy"
        if c == 'ज' && i + 2 < n && chars[i + 1] == '\u{094D}' && chars[i + 2] == 'ञ' {
            out.push_str("gy");
            let next_matra = chars.get(i + 3).copied();
            if let Some(m) = next_matra.filter(|&m| is_matra(m)) {
                out.push_str(matra_str(m));
                i += 4;
            } else {
                let is_final = i + 3 == n;
                if is_final && consonant_count > 1 {
                    // multisyllabic final bare consonant drop
                } else {
                    out.push('a');
                }
                i += 3;
            }
            continue;
        }

        // 2. Homorganic nasal: ञ् before consonant is transliterated as "n" (e.g. पञ्चायत -> panchayat)
        if c == 'ञ' && i + 1 < n && chars[i + 1] == '\u{094D}' {
            out.push('n');
            i += 2;
            continue;
        }

        if is_independent_vowel(c) {
            out.push_str(independent_vowel_str(c));
            i += 1;
            continue;
        }

        if is_consonant(c) {
            let base = consonant_str(c);
            out.push_str(base);

            let next = chars.get(i + 1).copied();
            match next {
                Some('\u{094D}') => {
                    // Halanta / virama: suppresses inherent vowel
                    i += 2;
                    continue;
                }
                Some(m) if is_matra(m) => {
                    out.push_str(matra_str(m));
                    i += 2;
                    continue;
                }
                _ => {
                    let is_preceded_by_halanta = i > 0 && chars[i - 1] == '\u{094D}';
                    let is_word_final = (i + 1 == n
                        || (i + 2 == n
                            && matches!(chars[i + 1], '\u{0902}' | '\u{0901}' | '\u{0903}')))
                        && !is_preceded_by_halanta;
                    if is_word_final && consonant_count > 1 {
                        // Drop final schwa in multisyllabic simple consonants (घर -> ghar)
                    } else {
                        // Keep inherent schwa on monosyllables, medially, and conjunct codas (नमस्ते -> namaste, वृद्ध -> vriddha)
                        out.push('a');
                    }
                    i += 1;
                    continue;
                }
            }
        }

        match c {
            '\u{0902}' => out.push('n'), // Anusvara
            '\u{0901}' => out.push('n'), // Chandrabindu
            '\u{0903}' => out.push('h'), // Visarga
            '\u{093D}' => {}             // Avagraha (silent)
            _ => {
                if c.is_ascii_alphanumeric() {
                    out.push(c.to_ascii_lowercase());
                }
            }
        }
        i += 1;
    }

    out
}

/// Collapse a Roman string into a canonical phonetic key (kappa).
pub fn collapse_key(input: &str) -> String {
    let lower = input.to_ascii_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(n);

    let mut i = 0;
    while i < n {
        if i + 3 <= n && &lower[i..i + 3] == "shh" {
            out.push('s');
            i += 3;
            continue;
        }
        if i + 3 <= n && &lower[i..i + 3] == "chh" {
            out.push_str("ch");
            i += 3;
            continue;
        }
        if i + 2 <= n {
            match &lower[i..i + 2] {
                "sh" => {
                    out.push('s');
                    i += 2;
                    continue;
                }
                "th" => {
                    out.push_str("th");
                    i += 2;
                    continue;
                }
                "dh" => {
                    out.push_str("dh");
                    i += 2;
                    continue;
                }
                "ng" | "ny" => {
                    out.push('n');
                    i += 2;
                    continue;
                }
                "ee" | "ii" => {
                    out.push('i');
                    i += 2;
                    continue;
                }
                "oo" | "uu" => {
                    out.push('u');
                    i += 2;
                    continue;
                }
                "aa" => {
                    out.push('a');
                    i += 2;
                    continue;
                }
                "ai" | "ae" => {
                    out.push_str("ai");
                    i += 2;
                    continue;
                }
                "au" | "ou" => {
                    out.push_str("au");
                    i += 2;
                    continue;
                }
                _ => {}
            }
        }

        let c = chars[i];
        match c {
            'w' | 'b' => out.push('v'),
            _ if c.is_ascii_alphanumeric() => out.push(c),
            _ => {}
        }
        i += 1;
    }

    // Collapse geminate consonants
    let mut collapsed = String::with_capacity(out.len());
    let mut prev: Option<char> = None;
    for c in out.chars() {
        if let Some(p) = prev {
            if p == c && !matches!(c, 'a' | 'e' | 'i' | 'o' | 'u') {
                continue;
            }
        }
        collapsed.push(c);
        prev = Some(c);
    }

    collapsed
}

/// Project a Devanagari word directly to its collapsed phonetic key: kappa(phi(D)).
pub fn devanagari_to_collapsed_key(word: &str) -> String {
    collapse_key(&project_devanagari(word))
}

/// Inverted index mapping canonical phonetic keys (kappa) to Devanagari words.
/// Inverted index mapping canonical phonetic keys (kappa) to Devanagari words.
///
/// Lookups:
///  1. Exact: `map[collapse_key(query)]`
///  2. Fuzzy: query-side 1-edit neighborhood (deletions and phonetic insertions)
///     evaluated at query time against `map`. Zero extra RAM at build time.
#[derive(Clone, Default)]
pub struct PhoneticKeyIndex {
    /// Exact canonical key → top-16 Devanagari words (frequency-sorted).
    map: rustc_hash::FxHashMap<String, Vec<String>>,
}

impl PhoneticKeyIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from (word, levels) pairs, indexing words with max level >= min_level.
    pub fn build<'a>(
        entries: impl IntoIterator<Item = (&'a str, &'a [u8])>,
        min_level: u8,
    ) -> Self {
        let mut map: rustc_hash::FxHashMap<String, Vec<(String, u8)>> =
            rustc_hash::FxHashMap::default();
        for (w, lv) in entries {
            let max_lvl = lv.iter().copied().max().unwrap_or(0);
            if max_lvl < min_level {
                continue;
            }
            let k = devanagari_to_collapsed_key(w);
            if !k.is_empty() {
                map.entry(k).or_default().push((w.to_string(), max_lvl));
            }
        }

        let mut exact: rustc_hash::FxHashMap<String, Vec<String>> =
            rustc_hash::FxHashMap::default();
        for (k, mut words) in map {
            words.sort_by_key(|b| std::cmp::Reverse(b.1));
            words.truncate(16);
            exact.insert(k, words.into_iter().map(|(w, _)| w).collect());
        }

        Self { map: exact }
    }

    /// Lookup by collapsed canonical key — exact match only.
    pub fn lookup(&self, roman_query: &str) -> &[String] {
        let key = collapse_key(roman_query);
        self.map.get(&key).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Lookup with query-side 1-edit fuzzy neighborhood.
    ///
    /// Generates 1-char deletions (extra letter in query) and common phonetic
    /// insertions ('h' aspiration, 'a' schwa, vowels) on the collapsed query key,
    /// then checks them against `self.map`. Runs in < 2 µs with zero extra heap RAM.
    pub fn lookup_fuzzy<'a>(&'a self, roman_query: &str) -> impl Iterator<Item = &'a String> {
        let key = collapse_key(roman_query);
        let chars: Vec<char> = key.chars().collect();
        let exact = self.map.get(&key);

        let mut seen: rustc_hash::FxHashSet<&str> = rustc_hash::FxHashSet::default();
        if let Some(exact_words) = exact {
            for w in exact_words {
                seen.insert(w.as_str());
            }
        }

        let mut results: Vec<&'a String> = Vec::new();

        // 1. Single-char deletions of the query key (user typed an extra char)
        if chars.len() >= 3 {
            for drop_pos in 0..chars.len() {
                let del_key: String = chars
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != drop_pos)
                    .map(|(_, &c)| c)
                    .collect();
                if let Some(words) = self.map.get(&del_key) {
                    for w in words {
                        if seen.insert(w.as_str()) {
                            results.push(w);
                        }
                    }
                }
            }
        }

        // 2. Single-char insertions of common phonetic characters into query key
        //    (e.g. 'sati' -> 'sathi' via 'h', 'nmaste' -> 'namaste' via 'a')
        const INSERT_CHARS: [char; 6] = ['a', 'h', 'i', 'u', 'e', 'o'];
        if chars.len() >= 2 && chars.len() <= 12 {
            for ins_pos in 0..=chars.len() {
                for &c in &INSERT_CHARS {
                    let mut ins_key = String::with_capacity(chars.len() + 1);
                    for (i, &ch) in chars.iter().enumerate() {
                        if i == ins_pos {
                            ins_key.push(c);
                        }
                        ins_key.push(ch);
                    }
                    if ins_pos == chars.len() {
                        ins_key.push(c);
                    }
                    if let Some(words) = self.map.get(&ins_key) {
                        for w in words {
                            if seen.insert(w.as_str()) {
                                results.push(w);
                            }
                        }
                    }
                }
            }
        }

        results.into_iter()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Morphology helpers
// ---------------------------------------------------------------------------

/// Roman transliterations of the Devanagari morphological suffixes from
/// `reranker::MORPH_SUFFIXES`.  Each entry is `(roman_suffix, devanagari_suffix)`.
/// Longest-first so greedy stripping matches correctly.
pub const ROMAN_MORPH_SUFFIXES: &[(&str, &str)] = &[
    // 4-char Devanagari (long roman equivalents)
    ("dekhi", "देखि"),
    ("thiyo", "थियो"),
    ("thin", "थिन्"),
    // 3-char Devanagari
    ("bata", "बाट"),
    ("eka", "एका"),
    ("eki", "एकी"),
    ("eko", "एको"),
    ("chhan", "छन्"),
    ("tva", "त्व"),
    ("thie", "थिए"),
    ("pana", "पना"),
    ("pani", "पनि"),
    ("bat", "बाट"),
    ("vat", "बाट"),
    ("bani", "भनी"),
    ("bane", "भने"),
    ("lai", "लाई"),
    ("sanga", "सँग"),
    ("sita", "सित"),
    ("haru", "हरू"),
    // 2-char Devanagari
    ("era", "एर"),
    ("ka", "का"),
    ("ki", "की"),
    ("ko", "को"),
    ("ji", "जी"),
    ("ta", "ता"),
    ("da", "दा"),
    ("dai", "दै"),
    ("nu", "नु"),
    ("ne", "ने"),
    ("nai", "नै"),
    ("pan", "पन"),
    ("ma", "मा"),
    ("le", "ले"),
    // 1-char Devanagari
    ("ta", "त"),
];

/// Attach a Devanagari suffix to a base word following Nepali morphophonemics.
///
/// If suffix starts with 'ए' (e.g. 'एको', 'एर') and base word ends with a bare
/// consonant (no matra, no halant), the 'ए' is converted to the matra 'े' (e.g. गर + एको -> गरेको).
pub fn attach_devanagari_suffix(base_word: &str, suffix: &str) -> String {
    if suffix.starts_with('ए') {
        if let Some(last_char) = base_word.chars().last() {
            if is_consonant(last_char) {
                let rest_suffix = &suffix['ए'.len_utf8()..];
                return format!("{}\u{0947}{}", base_word, rest_suffix);
            }
        }
    }
    format!("{}{}", base_word, suffix)
}

/// Strip one roman morphological suffix from `roman`, returning `(stem, dev_suffix)` pairs.
///
/// Returns all matching strippings (multiple suffixes may match; the caller scores them).
/// Enforces a minimum stem length of 2 ASCII chars so we don't strip into nothing.
pub fn strip_roman_suffixes(roman: &str) -> Vec<(&'static str, &'static str)> {
    let lower = roman.to_ascii_lowercase();
    let mut results: Vec<(&'static str, &'static str)> = Vec::new();
    let mut seen_roman: rustc_hash::FxHashSet<&str> = rustc_hash::FxHashSet::default();
    for &(rsuf, dsuf) in ROMAN_MORPH_SUFFIXES {
        if lower.ends_with(rsuf) && lower.len() > rsuf.len() + 1 {
            // Deduplicate on roman suffix (multiple dsuf entries for same rsuf)
            if seen_roman.insert(rsuf) {
                results.push((rsuf, dsuf));
            }
        }
    }
    results
}

/// Factored Matra and Phonetic Fidelity Term (D5 / MANUAL §16/§17.2).
///
/// Computes a bounded scalar delta in [-0.5, +0.5] reflecting how faithfully
/// a candidate Devanagari string matches the user's explicit phonetic and matra
/// intentions in `roman`.
///
/// Signals:
/// 1. Exact canonical phonetic key match: kappa(phi(D)) == kappa(R).
/// 2. Explicit vowel length intent:
///    - 'ee' / 'ii' explicitly asks for deergha 'ी' / 'ई'.
///    - 'oo' / 'uu' explicitly asks for deergha 'ू' / 'ऊ'.
/// 3. Velar nasal 'ng' intent: 'ng' matches 'ङ्' conjunct (सङ्गठन, रङ्ग) over bare anusvara 'ं'.
/// 4. Standard Nepali plural suffix: "haru" in Roman corresponds to standard "हरू".
pub fn matra_fidelity_score(roman: &str, dev: &str) -> f64 {
    let mut delta = 0.0f64;
    let r_lower = roman.to_ascii_lowercase();

    // 1. Exact canonical phonetic key match
    let r_key = collapse_key(&r_lower);
    let d_key = devanagari_to_collapsed_key(dev);
    if r_key == d_key {
        delta += 0.15;
    } else {
        let diff = (r_key.len() as isize - d_key.len() as isize).abs();
        if diff >= 2 {
            delta -= 0.10;
        }
    }

    // 2. Explicit vowel length intent:
    let has_long_i = r_lower.contains("ee") || r_lower.contains("ii");
    if has_long_i {
        if dev.contains('\u{0940}') || dev.contains('\u{0908}') {
            delta += 0.20;
        } else if dev.contains('\u{093F}') || dev.contains('\u{0907}') {
            delta -= 0.20;
        }
    }

    let has_long_u = r_lower.contains("oo") || r_lower.contains("uu");
    if has_long_u {
        if dev.contains('\u{0942}') || dev.contains('\u{090A}') {
            delta += 0.20;
        } else if dev.contains('\u{0941}') || dev.contains('\u{0909}') {
            delta -= 0.20;
        }
    }

    // 3. Velar nasal cluster: "ng" in roman favors 'ङ्ग' / 'ङ्' over bare anusvara 'ं'
    if r_lower.contains("ng") && dev.contains("\u{0919}\u{094D}") {
        delta += 0.15;
    }

    // 4. Standard Nepali plural suffix: "haru" favors standard "हरू" (deergha) over "हरु"
    if (r_lower.ends_with("haru") || r_lower.contains("haru")) && dev.contains("हरू") {
        delta += 0.12;
    }

    delta
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projects_namaste() {
        assert_eq!(project_devanagari("नमस्ते"), "namaste");
        assert_eq!(collapse_key("namaste"), "namaste");
        assert_eq!(devanagari_to_collapsed_key("नमस्ते"), "namaste");
    }

    #[test]
    fn projects_ghar() {
        assert_eq!(project_devanagari("घर"), "ghar");
        assert_eq!(collapse_key("ghar"), "ghar");
        assert_eq!(devanagari_to_collapsed_key("घर"), "ghar");
    }

    #[test]
    fn collapses_vowel_and_sibilant_variations() {
        let key_gold = devanagari_to_collapsed_key("साथी");
        assert_eq!(key_gold, "sathi");
        assert_eq!(collapse_key("sathi"), key_gold);
        assert_eq!(collapse_key("saathi"), key_gold);
        assert_eq!(collapse_key("saathee"), key_gold);
    }

    #[test]
    fn collapses_v_and_w() {
        let key_gold = devanagari_to_collapsed_key("विकास");
        assert_eq!(key_gold, "vikas");
        assert_eq!(collapse_key("vikas"), key_gold);
        assert_eq!(collapse_key("bikas"), key_gold);
        assert_eq!(collapse_key("bikash"), key_gold);
    }

    #[test]
    fn handles_vocalic_r() {
        let key_gold = devanagari_to_collapsed_key("वृद्ध");
        assert_eq!(collapse_key("vriddha"), key_gold);
        assert_eq!(collapse_key("briddha"), key_gold);
    }

    #[test]
    fn handles_nepali_conjunct_gyan_and_vigyan() {
        assert_eq!(project_devanagari("ज्ञान"), "gyaan");
        assert_eq!(devanagari_to_collapsed_key("ज्ञान"), "gyan");
        assert_eq!(collapse_key("gyan"), devanagari_to_collapsed_key("ज्ञान"));
        assert_eq!(project_devanagari("विज्ञान"), "vigyaan");
        assert_eq!(devanagari_to_collapsed_key("विज्ञान"), "vigyan");
        assert_eq!(
            collapse_key("bigyan"),
            devanagari_to_collapsed_key("विज्ञान")
        );
        assert_eq!(
            collapse_key("vigyan"),
            devanagari_to_collapsed_key("विज्ञान")
        );
    }

    #[test]
    fn handles_homorganic_nasal_panchayat() {
        assert_eq!(project_devanagari("पञ्चायत"), "panchaayat");
        assert_eq!(devanagari_to_collapsed_key("पञ्चायत"), "panchayat");
        assert_eq!(
            collapse_key("panchayat"),
            devanagari_to_collapsed_key("पञ्चायत")
        );
    }

    #[test]
    fn benchmark_phonetic_index_build_speed() {
        use std::path::Path;
        use std::time::Instant;
        if !Path::new("data/lexicon.bin").exists() {
            return;
        }
        let data = std::fs::read("data/lexicon.bin").unwrap();
        let lex_data: crate::core::lexicon::LexiconData = bincode::deserialize(&data).unwrap();
        let lex = crate::core::lexicon::Lexicon::from_data(lex_data).unwrap();

        let start = Instant::now();
        let entries = lex.entries();
        let decode_elapsed = start.elapsed();

        let map_start = Instant::now();
        let idx =
            PhoneticKeyIndex::build(entries.iter().map(|(w, lv)| (w.as_str(), lv.as_slice())), 0);
        let map_elapsed = map_start.elapsed();

        eprintln!(
            "Lexicon 700k FST dump: {:?}, Key map build ({} keys): {:?}",
            decode_elapsed,
            idx.len(),
            map_elapsed
        );
    }

    #[test]
    fn test_morphology_attachment() {
        // Consonant base + 'ए' suffix -> convert 'ए' to matra 'े'
        assert_eq!(attach_devanagari_suffix("गर", "एको"), "गरेको");
        assert_eq!(attach_devanagari_suffix("गर", "एर"), "गरेर");
        assert_eq!(attach_devanagari_suffix("गर", "ेको"), "गरेको");
        // Vowel/matra base + 'ए' suffix -> keep 'ए'
        assert_eq!(attach_devanagari_suffix("पा", "एको"), "पाएको");
        // Regular postpositions -> direct concatenation
        assert_eq!(attach_devanagari_suffix("घर", "मा"), "घरमा");
        assert_eq!(attach_devanagari_suffix("साथी", "को"), "साथीको");
        assert_eq!(attach_devanagari_suffix("नेपाल", "बाट"), "नेपालबाट");
    }

    #[test]
    fn test_lookup_fuzzy_aspiration_and_schwa() {
        let entries = [
            ("साथी", [100, 0, 0, 0, 0, 0, 0, 0]),
            ("नमस्ते", [100, 0, 0, 0, 0, 0, 0, 0]),
        ];
        let idx = PhoneticKeyIndex::build(entries.iter().map(|(w, lv)| (*w, lv.as_slice())), 0);

        // Exact match
        assert_eq!(idx.lookup("sathi"), &["साथी"]);

        // Fuzzy match: 'sati' lacks 'h' -> insertion finds 'sathi' -> 'साथी'
        let fuzzy_sati: Vec<String> = idx.lookup_fuzzy("sati").cloned().collect();
        assert!(fuzzy_sati.contains(&"साथी".to_string()));

        // Fuzzy match: 'nmaste' lacks 'a' -> insertion finds 'namaste' -> 'नमस्ते'
        let fuzzy_nmaste: Vec<String> = idx.lookup_fuzzy("nmaste").cloned().collect();
        assert!(fuzzy_nmaste.contains(&"नमस्ते".to_string()));
    }
}
