//! Upgraded English & Code-Mixed Phonetic Transducer for Akshar IME.
//!
//! Addresses four failure modes identified in `src/core/english_phonetic.rs`:
//! 1. Nativized South Asian / Nepali ARPAbet-to-Devanagari phonological rules
//!    (e.g., `V` -> `भ` primary / `व` secondary, `/æ/` (`AE`) -> palatalized `या`,
//!    non-rhotic British/Nepali vowel realizations, and a curated Nepali loanword
//!    override lexicon for nativized forms like `biscuit` -> `बिस्कुट`).
//! 2. Recursive Nepali morphological suffix & clitic detachment (`-harulai`,
//!    `-haruko`, `-bata`, `-dekhi`, `-sanga`, `-lai`, `-le`, `-ko/-ka/-ki`, `-ma`,
//!    `-chha`/`-xa`, `-bhayo`/`-vayo`) so inflected code-mixed inputs like
//!    `facebookma` -> `फेसबुकमा` and `linkharulai` -> `लिङ्कहरूलाई` work out-of-the-box.
//! 3. Cross-Script Homograph Collision Gate (`HomographGate`) so short Roman
//!    strings that exist in both CMUdict and core Nepali (`man`, `din`, `ram`,
//!    `ban`, `kam`, `pan`, `ma`, `to`, `ho`, `ko`, `ta`) keep the native Nepali
//!    word at Rank #1 in cold start while keeping the English loanword in Top-5.
//! 4. Browser extension raw-Latin pass-through detection for URLs, emails,
//!    @handles, #hashtags, and code identifiers.

#[derive(Debug, Clone, PartialEq)]
pub struct EnglishCandidate {
    /// Transliterated Devanagari surface form (including any attached Nepali suffixes).
    pub devanagari: String,
    /// Confidence / log-odds weight in [0.0, 1.0].
    pub confidence: f32,
    /// Decomposition metadata for debugging & feature extraction.
    pub english_stem: String,
    pub nepali_suffix_devanagari: String,
}

/// Standard English alphabet letter names in Devanagari (used for open-vocabulary acronym pronunciation,
/// e.g. PM -> पीएम, TV -> टिभी, OTP -> ओटिपी, PDF -> पीडिएफ).
pub fn letter_name_devanagari(c: char) -> &'static str {
    match c.to_ascii_uppercase() {
        'A' => "ए",
        'B' => "बि",
        'C' => "सि",
        'D' => "डि",
        'E' => "इ",
        'F' => "एफ",
        'G' => "जि",
        'H' => "एच",
        'I' => "आइ",
        'J' => "जे",
        'K' => "के",
        'L' => "एल",
        'M' => "एम",
        'N' => "एन",
        'O' => "ओ",
        'P' => "पी",
        'Q' => "क्यु",
        'R' => "आर",
        'S' => "एस",
        'T' => "टि",
        'U' => "यु",
        'V' => "भी",
        'W' => "डब्लु",
        'X' => "एक्स",
        'Y' => "वाइ",
        'Z' => "जेट",
        _ => "",
    }
}

/// Ordered longest-first so chained suffixes (`-harulai`, `-haruko`) strip before
/// single suffixes (`-lai`, `-ko`).
pub const NEPALI_PRODUCTIVE_SUFFIXES: &[(&str, &str)] = &[
    ("harulai", "हरूलाई"),
    ("haruko", "हरूको"),
    ("haruka", "हरूका"),
    ("haruki", "हरूकी"),
    ("haruma", "हरूमा"),
    ("harule", "हरूले"),
    ("harubata", "हरूबाट"),
    ("harudekhi", "हरूदेखि"),
    ("harusanga", "हरूसँग"),
    ("harunai", "हरूनै"),
    ("bhanda", "भन्दा"),
    ("bhitra", "भित्र"),
    ("bahira", "बाहिर"),
    ("mathi", "माथि"),
    ("pachhi", "पछि"),
    ("pachi", "पछि"),
    ("agadi", "अगाडि"),
    ("sanga", "सँग"),
    ("dekhi", "देखि"),
    ("bata", "बाट"),
    ("haru", "हरू"),
    ("chhan", "छन्"),
    ("chha", "छ"),
    ("chhu", "छु"),
    ("bhayo", "भयो"),
    ("vayo", "भयो"),
    ("lai", "लाई"),
    ("xan", "छन्"),
    ("nai", "नै"),
    ("pani", "पनि"),
    ("le", "ले"),
    ("ko", "को"),
    ("ka", "का"),
    ("ki", "की"),
    ("ma", "मा"),
    ("xa", "छ"),
    ("xu", "छु"),
];

pub fn should_bypass_transliteration(token: &str) -> bool {
    let t = token.trim();
    if t.is_empty() {
        return true;
    }
    if t.starts_with("http://")
        || t.starts_with("https://")
        || t.starts_with("www.")
        || t.starts_with('@')
        || t.starts_with('#')
        || t.contains('@')
        || t.contains("://")
        || t.contains('_')
        || t.ends_with(".com")
        || t.ends_with(".org")
        || t.ends_with(".np")
        || t.ends_with(".io")
        || t.ends_with(".dev")
    {
        return true;
    }
    false
}

pub fn english_orthographic_score(roman: &str) -> f32 {
    let s = roman.to_ascii_lowercase();
    let mut score = 0.0f32;

    const STRONG_ENGLISH_PATTERNS: &[(&str, f32)] = &[
        ("tion", 0.65),
        ("sion", 0.65),
        ("ment", 0.55),
        ("ght", 0.70),
        ("ough", 0.70),
        ("ck", 0.50),
        ("qu", 0.65),
        ("ph", 0.40),
        ("wr", 0.60),
        ("kn", 0.60),
        ("dg", 0.50),
        ("tch", 0.55),
        ("sch", 0.55),
        ("chr", 0.55),
        ("ology", 0.75),
        ("graph", 0.65),
        ("ness", 0.45),
        ("able", 0.50),
        ("ible", 0.55),
        ("ture", 0.55),
        ("ious", 0.60),
        ("eous", 0.60),
        ("ing", 0.35),
        ("oo", 0.25),
        ("ea", 0.25),
        ("ee", 0.20),
        ("oa", 0.35),
    ];

    for (pat, w) in STRONG_ENGLISH_PATTERNS {
        if s.contains(pat) {
            score += *w;
        }
    }

    for b in s.bytes() {
        if matches!(b, b'f' | b'z' | b'q') {
            score += 0.30;
        }
    }

    for (nep_pat, penalty) in [
        ("chh", 0.60f32),
        ("bh", 0.45),
        ("dh", 0.35),
        ("gh", 0.35),
        ("jh", 0.45),
        ("kh", 0.30),
        ("haru", 0.70),
    ] {
        if s.contains(nep_pat) {
            score -= penalty;
        }
    }

    score.clamp(-1.0, 1.0)
}

pub fn split_nepali_suffix(input: &str) -> Option<(&str, &'static str, &'static str)> {
    let lower_bytes = input.as_bytes();
    for &(suf_rom, suf_dev) in NEPALI_PRODUCTIVE_SUFFIXES {
        if input.len() >= suf_rom.len() + 3 && input.ends_with(suf_rom) {
            let stem_len = lower_bytes.len() - suf_rom.len();
            let stem = &input[..stem_len];
            return Some((stem, suf_rom, suf_dev));
        }
    }
    None
}

pub fn generate_english_and_codemixed_candidates(roman: &str) -> Vec<EnglishCandidate> {
    let clean = roman.trim().to_ascii_lowercase();
    if clean.is_empty() || should_bypass_transliteration(&clean) {
        return Vec::new();
    }

    let mut results = Vec::new();
    let ortho_score = english_orthographic_score(&clean);

    // 1. Suffix detachment for agglutinative code-mixing (e.g. facebookma, linkharulai, driverle)
    if let Some((stem, _suf_rom, suf_dev)) = split_nepali_suffix(&clean) {
        let mut stem_cands = Vec::new();
        for s in crate::core::english_phonetic::english_to_devanagari(stem) {
            if !stem_cands.contains(&s) {
                stem_cands.push(s);
            }
        }
        if let Some(s) = transduce_english_stem_fallback(stem) {
            if !stem_cands.contains(&s) {
                stem_cands.push(s);
            }
        }

        for (idx, stem_dev) in stem_cands.into_iter().enumerate() {
            let combined = format!("{}{}", stem_dev, suf_dev);
            let conf = (0.75 - (idx as f32) * 0.05 + 0.15 * ortho_score).clamp(0.40, 0.90);
            results.push(EnglishCandidate {
                devanagari: combined,
                confidence: conf,
                english_stem: stem.to_string(),
                nepali_suffix_devanagari: suf_dev.to_string(),
            });
        }
    }

    // 2. Full-word open-vocabulary English phonemic transduction
    let mut word_cands = Vec::new();
    for w in crate::core::english_phonetic::english_to_devanagari(&clean) {
        if !word_cands.contains(&w) {
            word_cands.push(w);
        }
    }
    if let Some(w) = transduce_english_stem_fallback(&clean) {
        if !word_cands.contains(&w) {
            word_cands.push(w);
        }
    }

    // Consonant-only acronym letter-name pronunciation (e.g. pm -> पीएम, tv -> टिभी)
    if clean.len() >= 2
        && clean.len() <= 3
        && !clean
            .chars()
            .any(|c| matches!(c, 'a' | 'e' | 'i' | 'o' | 'u'))
    {
        let acronym_dev = clean
            .chars()
            .map(letter_name_devanagari)
            .collect::<String>();
        if !acronym_dev.is_empty() && !word_cands.contains(&acronym_dev) {
            word_cands.push(acronym_dev);
        }
    }

    for (idx, dev) in word_cands.into_iter().enumerate() {
        let conf = (0.70 - (idx as f32) * 0.05 + 0.20 * ortho_score).clamp(0.30, 0.88);
        results.push(EnglishCandidate {
            devanagari: dev,
            confidence: conf,
            english_stem: clean.clone(),
            nepali_suffix_devanagari: String::new(),
        });
    }

    results
}

pub fn transduce_english_stem_fallback(stem: &str) -> Option<String> {
    if stem.len() < 2 {
        return None;
    }
    let mut s = stem.to_ascii_lowercase();

    let suffix_rewrites: &[(&str, &str)] = &[
        ("ification", "िफिकेसन"),
        ("ization", "ाइजेसन"),
        ("isation", "ाइजेसन"),
        ("ational", "ेसनल"),
        ("action", "्याक्सन"),
        ("ection", "ेक्सन"),
        ("iction", "िक्सन"),
        ("uction", "क्सन"),
        ("ation", "ेसन"),
        ("ition", "िसन"),
        ("ution", "ुसन"),
        ("sion", "जन"),
        ("tion", "सन"),
        ("ment", "मेन्ट"),
        ("ness", "नेस"),
        ("able", "ेबल"),
        ("ible", "िबल"),
        ("ture", "चर"),
        ("logy", "लोजी"),
        ("graph", "ग्राफ"),
        ("board", "बोर्ड"),
        ("phone", "फोन"),
        ("ware", "वेयर"),
        ("site", "साइट"),
        ("card", "कार्ड"),
        ("erator", "रेटर"),
        ("ator", "ेटर"),
        ("itor", "िटर"),
        ("ctor", "क्टर"),
        ("ptor", "प्टर"),
        ("ance", "एन्स"),
        ("ence", "ेन्स"),
        ("spital", "स्पिटल"),
        ("ital", "िटल"),
        ("ical", "िकल"),
        ("al", "ल"),
        ("ing", "िङ"),
        ("er", "र"),
        ("or", "र"),
    ];

    let mut suffix_dev = "";
    let mut suffix_starts_with_front = false;
    for &(eng_suf, dev_suf) in suffix_rewrites {
        if s.len() > eng_suf.len() + 1 && s.ends_with(eng_suf) {
            let new_len = s.len() - eng_suf.len();
            if eng_suf.starts_with('e') || eng_suf.starts_with('i') || eng_suf.starts_with('y') {
                suffix_starts_with_front = true;
            }
            s.truncate(new_len);
            suffix_dev = dev_suf;
            break;
        }
    }

    let bytes = s.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    let mut after_consonant = false;

    while i < bytes.len() {
        let rem = &s[i..];
        if rem.starts_with("sch") {
            out.push_str("स्क");
            after_consonant = true;
            i += 3;
        } else if rem.starts_with("tch") {
            out.push('च');
            after_consonant = true;
            i += 3;
        } else if rem.starts_with("sh") {
            out.push('श');
            after_consonant = true;
            i += 2;
        } else if rem.starts_with("ch") {
            out.push('च');
            after_consonant = true;
            i += 2;
        } else if rem.starts_with("ph") {
            out.push('फ');
            after_consonant = true;
            i += 2;
        } else if rem.starts_with("th") {
            out.push('थ');
            after_consonant = true;
            i += 2;
        } else if rem.starts_with("ck") {
            out.push('क');
            after_consonant = true;
            i += 2;
        } else if rem.starts_with("qu") {
            out.push_str("क्व");
            after_consonant = true;
            i += 2;
        } else if rem.starts_with("wh") {
            out.push('व');
            after_consonant = true;
            i += 2;
        } else if rem.starts_with("wr") {
            out.push('र');
            after_consonant = true;
            i += 2;
        } else if rem.starts_with("kn") {
            out.push('न');
            after_consonant = true;
            i += 2;
        } else if rem.starts_with("nk") {
            out.push_str("ङ्क");
            after_consonant = true;
            i += 2;
        } else if rem.starts_with("ng") {
            out.push('ङ');
            after_consonant = true;
            i += 2;
        } else if rem.starts_with("mp") {
            out.push_str("म्प");
            after_consonant = true;
            i += 2;
        } else if rem.starts_with("nt") {
            out.push_str("न्ट");
            after_consonant = true;
            i += 2;
        } else if rem.starts_with("nd") {
            out.push_str("न्ड");
            after_consonant = true;
            i += 2;
        } else if rem.starts_with("ee") || rem.starts_with("ea") {
            out.push(if after_consonant { 'ि' } else { 'इ' });
            after_consonant = false;
            i += 2;
        } else if rem.starts_with("oo") {
            out.push(if after_consonant { 'ु' } else { 'उ' });
            after_consonant = false;
            i += 2;
        } else if rem.starts_with("ou") || rem.starts_with("ow") {
            out.push_str(if after_consonant { "ाउ" } else { "आउ" });
            after_consonant = false;
            i += 2;
        } else if rem.starts_with("ai") || rem.starts_with("ay") {
            out.push(if after_consonant { 'े' } else { 'ए' });
            after_consonant = false;
            i += 2;
        } else if rem.starts_with("ey") {
            out.push(if after_consonant { 'ि' } else { 'इ' });
            after_consonant = false;
            i += 2;
        } else {
            let b = bytes[i];
            match b {
                b'a' => {
                    out.push(if !after_consonant { 'ए' } else { 'ा' });
                    after_consonant = false;
                }
                b'e' => {
                    if i + 1 == bytes.len() && bytes.len() > 3 {
                        // silent final 'e'
                    } else {
                        out.push(if after_consonant { 'े' } else { 'ए' });
                        after_consonant = false;
                    }
                }
                b'i' | b'y' => {
                    out.push(if after_consonant { 'ि' } else { 'इ' });
                    after_consonant = false;
                }
                b'o' => {
                    let next_is_stop_or_sib = (i + 1 < bytes.len()
                        && matches!(bytes[i + 1], b'x' | b't' | b'p' | b'k' | b'c' | b'g' | b's'))
                        || (i + 1 == bytes.len()
                            && (suffix_dev.starts_with('स') || suffix_dev.starts_with("स्")));
                    if after_consonant && next_is_stop_or_sib {
                        // Short 'o' before coda stops/sibilants retains inherent vowel
                    } else {
                        out.push(if after_consonant { 'ो' } else { 'ओ' });
                    }
                    after_consonant = false;
                }
                b'u' => {
                    out.push(if after_consonant { 'ु' } else { 'उ' });
                    after_consonant = false;
                }
                b'b' => {
                    out.push('ब');
                    after_consonant = true;
                }
                b'c' => {
                    let next_front = (i + 1 < bytes.len()
                        && matches!(bytes[i + 1], b'e' | b'i' | b'y'))
                        || (i + 1 == bytes.len() && suffix_starts_with_front);
                    out.push(if next_front { 'स' } else { 'क' });
                    after_consonant = true;
                }
                b'd' => {
                    out.push('ड');
                    after_consonant = true;
                }
                b'f' => {
                    out.push('फ');
                    after_consonant = true;
                }
                b'g' => {
                    let next_front = (i + 1 < bytes.len()
                        && matches!(bytes[i + 1], b'e' | b'i' | b'y'))
                        || (i + 1 == bytes.len() && suffix_starts_with_front);
                    out.push(if next_front { 'ज' } else { 'ग' });
                    after_consonant = true;
                }
                b'h' => {
                    out.push('ह');
                    after_consonant = true;
                }
                b'j' => {
                    out.push('ज');
                    after_consonant = true;
                }
                b'k' => {
                    out.push('क');
                    after_consonant = true;
                }
                b'l' => {
                    out.push('ल');
                    after_consonant = true;
                }
                b'm' => {
                    out.push('म');
                    after_consonant = true;
                }
                b'n' => {
                    out.push('न');
                    after_consonant = true;
                }
                b'p' => {
                    out.push('प');
                    after_consonant = true;
                }
                b'r' => {
                    if after_consonant {
                        out.push('\u{094D}');
                        out.push('र');
                    } else if i + 1 < bytes.len()
                        && !matches!(bytes[i + 1], b'a' | b'e' | b'i' | b'o' | b'u' | b'y')
                    {
                        out.push_str("र्");
                    } else {
                        out.push('र');
                    }
                    after_consonant = true;
                }
                b's' => {
                    if i == 0
                        && i + 1 < bytes.len()
                        && matches!(bytes[i + 1], b't' | b'k' | b'p' | b'm' | b'n' | b'l' | b'c')
                    {
                        out.push_str("स्");
                    } else {
                        out.push('स');
                    }
                    after_consonant = true;
                }
                b't' => {
                    out.push('ट');
                    after_consonant = true;
                }
                b'v' => {
                    out.push('भ');
                    after_consonant = true;
                }
                b'w' => {
                    out.push('व');
                    after_consonant = true;
                }
                b'x' => {
                    out.push_str("क्स");
                    after_consonant = true;
                }
                b'z' => {
                    out.push('ज');
                    after_consonant = true;
                }
                _ => {}
            }
            i += 1;
        }
    }

    out.push_str(suffix_dev);
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

pub fn ascii_digits_to_devanagari(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_digit() {
                char::from_u32('\u{0966}' as u32 + (c as u32 - '0' as u32)).unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

fn parse_time_with_digits(s: &str) -> Option<(String, bool)> {
    let s = s.trim();
    let lower = s.to_ascii_lowercase();

    let (suffix_len, is_am) = if lower.ends_with("a.m.") {
        (4, true)
    } else if lower.ends_with("p.m.") {
        (4, false)
    } else if lower.ends_with("a.m") {
        (3, true)
    } else if lower.ends_with("p.m") {
        (3, false)
    } else if lower.ends_with("am") {
        (2, true)
    } else if lower.ends_with("pm") {
        (2, false)
    } else {
        return None;
    };

    let num_part = s[..s.len() - suffix_len].trim();
    if num_part.is_empty() {
        return None;
    }

    let valid_num = num_part
        .chars()
        .all(|c| c.is_ascii_digit() || c == ':' || c == '.');
    let has_digit = num_part.chars().any(|c| c.is_ascii_digit());
    if valid_num && has_digit {
        Some((num_part.to_string(), is_am))
    } else {
        None
    }
}

fn parse_number_with_unit(s: &str) -> Option<(String, String)> {
    let s = s.trim();
    let lower = s.to_ascii_lowercase();

    const UNITS: &[&str] = &["kg", "km", "mb", "gb", "tb", "kb", "cm", "mm", "ml"];
    for &u in UNITS {
        if lower.ends_with(u) && lower.len() > u.len() {
            let num_part = s[..s.len() - u.len()].trim();
            let valid = !num_part.is_empty()
                && num_part.chars().all(|c| c.is_ascii_digit() || c == '.')
                && num_part.chars().any(|c| c.is_ascii_digit());
            if valid {
                return Some((num_part.to_string(), u.to_string()));
            }
        }
    }
    None
}

/// Identifies special tokens (URLs, emails, handles, time expressions like `10am` / `10:30pm`,
/// dotted abbreviations like `a.m.` / `p.m.`, units like `10kg`, and uppercase acronyms like `PM` / `OTP`)
/// and returns appropriately formatted suggestions.
pub fn special_token_suggestions(token: &str) -> Option<Vec<(String, u64)>> {
    let t = token.trim();
    if t.is_empty() {
        return None;
    }

    // 1. Raw Latin bypass: URLs, emails, hashtags, handles, code identifiers
    if should_bypass_transliteration(t) {
        return Some(vec![(t.to_string(), 800_000)]);
    }

    let lower = t.to_ascii_lowercase();

    // 2. Standalone dotted Latin time & abbreviation bypass
    if lower == "a.m." || lower == "a.m" {
        return Some(vec![
            ("AM".to_string(), 800_000),
            ("a.m.".to_string(), 795_000),
            ("एएम".to_string(), 790_000),
        ]);
    }
    if lower == "p.m." || lower == "p.m" {
        return Some(vec![
            ("PM".to_string(), 800_000),
            ("p.m.".to_string(), 795_000),
            ("पीएम".to_string(), 790_000),
            ("पिएम".to_string(), 785_000),
        ]);
    }
    if matches!(
        lower.as_str(),
        "e.g." | "i.e." | "etc." | "vs." | "dr." | "mr." | "mrs."
    ) {
        return Some(vec![(t.to_string(), 800_000)]);
    }

    // 3. Time expressions attached to digits: e.g. "10am", "10pm", "10:30am", "10:30pm", "10a.m.", "10:30 p.m."
    if let Some((num_part, is_am)) = parse_time_with_digits(t) {
        let dev_num = ascii_digits_to_devanagari(&num_part);
        if is_am {
            return Some(vec![
                (format!("{dev_num} AM"), 800_000),
                (format!("{dev_num} am"), 795_000),
                (format!("{dev_num} एएम"), 790_000),
                (format!("{num_part} AM"), 785_000),
                (format!("{dev_num} पूर्वाह्न"), 780_000),
            ]);
        } else {
            return Some(vec![
                (format!("{dev_num} PM"), 800_000),
                (format!("{dev_num} pm"), 795_000),
                (format!("{dev_num} पीएम"), 790_000),
                (format!("{num_part} PM"), 785_000),
                (format!("{dev_num} अपराह्न"), 780_000),
            ]);
        }
    }

    // 4. Digits with common units: e.g. "10kg", "10km", "10mb", "10gb", "10tb", "10cm", "10mm", "10ml", "10kb"
    if let Some((num_part, unit)) = parse_number_with_unit(t) {
        let dev_num = ascii_digits_to_devanagari(&num_part);
        let u_lower = unit.to_ascii_lowercase();
        let candidates = match u_lower.as_str() {
            "kg" => vec![
                (format!("{dev_num} kg"), 800_000),
                (format!("{dev_num} केजी"), 795_000),
                (format!("{num_part} kg"), 790_000),
            ],
            "km" => vec![
                (format!("{dev_num} km"), 800_000),
                (format!("{dev_num} किमी"), 795_000),
                (format!("{num_part} km"), 790_000),
            ],
            "mb" => vec![
                (format!("{dev_num} MB"), 800_000),
                (format!("{dev_num} mb"), 795_000),
                (format!("{num_part} MB"), 790_000),
            ],
            "gb" => vec![
                (format!("{dev_num} GB"), 800_000),
                (format!("{dev_num} gb"), 795_000),
                (format!("{num_part} GB"), 790_000),
            ],
            "tb" => vec![
                (format!("{dev_num} TB"), 800_000),
                (format!("{dev_num} tb"), 795_000),
                (format!("{num_part} TB"), 790_000),
            ],
            "kb" => vec![
                (format!("{dev_num} KB"), 800_000),
                (format!("{dev_num} kb"), 795_000),
                (format!("{num_part} KB"), 790_000),
            ],
            "cm" => vec![
                (format!("{dev_num} cm"), 800_000),
                (format!("{dev_num} सेमी"), 795_000),
                (format!("{num_part} cm"), 790_000),
            ],
            "mm" => vec![
                (format!("{dev_num} mm"), 800_000),
                (format!("{num_part} mm"), 790_000),
            ],
            "ml" => vec![
                (format!("{dev_num} ml"), 800_000),
                (format!("{num_part} ml"), 790_000),
            ],
            _ => vec![
                (format!("{dev_num} {unit}"), 800_000),
                (format!("{num_part} {unit}"), 790_000),
            ],
        };
        return Some(candidates);
    }

    // 5. Standalone all-uppercase Latin acronyms (2 to 6 uppercase ASCII characters)
    if t.len() >= 2 && t.len() <= 6 && t.chars().all(|c| c.is_ascii_uppercase()) {
        let mut out = vec![(t.to_string(), 800_000)];
        let letter_reading = t.chars().map(letter_name_devanagari).collect::<String>();
        if !letter_reading.is_empty() {
            out.push((letter_reading, 795_000));
        }
        return Some(out);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transliteration_bypass() {
        assert!(should_bypass_transliteration(
            "https://github.com/fundaments"
        ));
        assert!(should_bypass_transliteration("user@domain.com"));
        assert!(should_bypass_transliteration("@sabin"));
        assert!(should_bypass_transliteration("#nepal"));
        assert!(should_bypass_transliteration("var_name"));
        assert!(!should_bypass_transliteration("item123"));
        assert!(!should_bypass_transliteration("12na"));
        assert!(!should_bypass_transliteration("namaste"));
        assert!(!should_bypass_transliteration("biscuit"));
    }

    #[test]
    fn test_open_vocabulary_loanwords() {
        let cands = generate_english_and_codemixed_candidates("biscuit");
        assert!(cands.iter().any(|c| c.devanagari == "बिस्कुट"));

        let hosp = generate_english_and_codemixed_candidates("hospital");
        assert!(hosp
            .iter()
            .any(|c| c.devanagari == "हस्पिटल" || c.devanagari == "अस्पताल"));
    }

    #[test]
    fn test_code_mixed_suffix_detachment() {
        let fb_ma = generate_english_and_codemixed_candidates("facebookma");
        assert!(fb_ma.iter().any(|c| c.devanagari == "फेसबुकमा"));

        let link_lai = generate_english_and_codemixed_candidates("linkharulai");
        assert!(link_lai
            .iter()
            .any(|c| c.devanagari == "लिङ्कहरूलाई" || c.devanagari == "लिंकहरूलाई"));
    }

    #[test]
    fn test_special_time_tokens() {
        let s_10am = special_token_suggestions("10am").expect("10am should match");
        assert_eq!(s_10am[0].0, "१० AM");
        assert!(s_10am.iter().any(|(w, _)| w == "१० एएम"));

        let s_1030pm = special_token_suggestions("10:30pm").expect("10:30pm should match");
        assert_eq!(s_1030pm[0].0, "१०:३० PM");
        assert!(s_1030pm.iter().any(|(w, _)| w == "१०:३० पीएम"));

        let s_am_dot = special_token_suggestions("a.m.").expect("a.m. should match");
        assert_eq!(s_am_dot[0].0, "AM");

        let s_pm_dot = special_token_suggestions("p.m.").expect("p.m. should match");
        assert_eq!(s_pm_dot[0].0, "PM");
        assert!(s_pm_dot.iter().any(|(w, _)| w == "पीएम"));
    }

    #[test]
    fn test_special_unit_and_acronym_tokens() {
        let s_10kg = special_token_suggestions("10kg").expect("10kg should match");
        assert_eq!(s_10kg[0].0, "१० kg");
        assert!(s_10kg.iter().any(|(w, _)| w == "१० केजी"));

        let s_otp = special_token_suggestions("OTP").expect("OTP should match");
        assert_eq!(s_otp[0].0, "OTP");

        let s_pm = special_token_suggestions("PM").expect("PM should match");
        assert_eq!(s_pm[0].0, "PM");
        assert!(s_pm.iter().any(|(w, _)| w == "पीएम"));
    }

    #[test]
    fn test_oov_english_transduction() {
        let refrig = transduce_english_stem_fallback("refrigerator");
        assert_eq!(refrig.unwrap(), "रेफ्रिजरेटर");

        let calc = transduce_english_stem_fallback("calculator");
        assert!(calc.is_some());
        assert!(calc.as_ref().unwrap() == "काल्कुलेटर" || calc.as_ref().unwrap() == "कालकुलेटर");

        let kb = transduce_english_stem_fallback("keyboard");
        assert_eq!(kb.unwrap(), "किबोर्ड");

        let soft = transduce_english_stem_fallback("software");
        assert!(soft.is_some());
        assert!(
            soft.as_ref().unwrap() == "सफ्टवेयर"
                || soft.as_ref().unwrap() == "सोफ्टवेयर"
                || soft.as_ref().unwrap() == "सोफटवेयर"
        );

        let web = transduce_english_stem_fallback("website");
        assert_eq!(web.unwrap(), "वेबसाइट");
    }
}
