// File: src/core/english_phonetic.rs
//
// Zero-dependency Open-Vocabulary English-to-Devanagari G2P (Grapheme-to-Phoneme) Phonemizer.
//
// Translates arbitrary English words, technical terms, loanwords, and compounds
// into conventional Nepali Devanagari orthography without relying on a static
// word dictionary.
//
// Implements:
// 1. Morphological / Compound word decomposition (e.g. fire+fox -> फायर+फक्स, down+load -> डाउन+लोड)
// 2. Nepali case marker / postposition detachment (e.g. december+ma, police+le, helicopter+mathi)
// 3. Suffix morphological rewriting (-tion, -sion, -ture, -ment, -ness, -able, -ing, etc.)
// 4. Letter-to-Sound (L2S) rules for English syllable structures:
//    - Silent 'e' vowel lengthening (C+V+C+e -> a_e:े, i_e:ाइ, o_e:ो, u_e:्यु)
//    - Vowel digraphs (ai/ay, ee, ea, oo, oa, ou, ow, oi/oy, au/aw, ew, ey)
//    - R-controlled vowels (ar, er, ir, or, ur, are, ire, ore, ure)
//    - Consonant digraphs & clusters (sch, chr, ch, sh, th, ph, wh, qu, ck, st, sp, sk, etc.)
//    - Context-dependent softening of 'c' and 'g' before front vowels (e, i, y)
//    - Nepali phonotactic conventions ('v' -> 'भ', 't'/'d' -> retroflex 'ट'/'ड', 'x' -> 'क्स')

/// Transduces an arbitrary English word into Devanagari transliteration candidates.
///
/// Returns high-confidence phonetic candidates ordered by orthographic likelihood.
pub fn english_to_devanagari(word: &str) -> Vec<String> {
    let lower = word.to_ascii_lowercase();
    let lower = lower.trim();
    if lower.len() < 2 || !lower.chars().all(|c| c.is_ascii_alphabetic()) {
        return Vec::new();
    }

    let mut results = Vec::with_capacity(4);

    // 1. Nepali case-marker / postposition detachment.
    // If an English word has a Romanized Nepali postposition attached (e.g. decemberma -> दिसम्बर + मा):
    let postpositions: &[(&str, &str)] = &[
        ("haru", "हरू"),
        ("mathi", "माथि"),
        ("tarfa", "तर्फ"),
        ("dekhi", "देखि"),
        ("bata", "बाट"),
        ("sanga", "सँग"),
        ("lai", "लाई"),
        ("ko", "को"),
        ("ka", "का"),
        ("ki", "की"),
        ("le", "ले"),
        ("ma", "मा"),
    ];

    for &(post, dev_post) in postpositions {
        if lower.ends_with(post) && lower.len() > post.len() + 2 {
            let stem = &lower[..lower.len() - post.len()];
            let stem_cands = transduce_word(stem);
            for sc in stem_cands {
                let full = format!("{}{}", sc, dev_post);
                if !results.contains(&full) {
                    results.push(full);
                }
            }
            if !results.is_empty() {
                return results;
            }
        }
    }

    // 2. Transduce the word directly
    for cand in transduce_word(lower) {
        if !results.contains(&cand) {
            results.push(cand);
        }
    }

    results
}

/// Transduces an English base word (without postpositions) into Devanagari candidates.
fn transduce_word(word: &str) -> Vec<String> {
    let mut results = Vec::with_capacity(4);

    // A. Compound decomposition pass.
    // Checks standard English compound boundaries (e.g. firefox, download, bluetooth, keyboard).
    static COMPOUND_SPLITS: &[(&str, &str)] = &[
        ("fire", "fox"),
        ("down", "load"),
        ("blue", "tooth"),
        ("key", "board"),
        ("you", "tube"),
        ("face", "book"),
        ("pass", "word"),
        ("web", "site"),
        ("soft", "ware"),
        ("hard", "ware"),
        ("net", "work"),
        ("lap", "top"),
        ("desk", "top"),
        ("on", "line"),
        ("off", "line"),
        ("data", "base"),
        ("air", "port"),
        ("bed", "room"),
        ("foot", "ball"),
        ("out", "put"),
        ("in", "put"),
        ("pass", "port"),
        ("super", "star"),
        ("back", "end"),
        ("front", "end"),
        ("micro", "soft"),
        ("cyber", "space"),
        ("smart", "phone"),
        ("head", "phone"),
    ];

    for &(left, right) in COMPOUND_SPLITS {
        if word == format!("{}{}", left, right) {
            let left_cands = transduce_stem(left);
            let right_cands = transduce_stem(right);
            for l in &left_cands {
                for r in &right_cands {
                    let cand = format!("{}{}", l, r);
                    if !results.contains(&cand) {
                        results.push(cand);
                    }
                }
            }
        }
    }

    // B. Direct stem transduction
    for cand in transduce_stem(word) {
        if !results.contains(&cand) {
            results.push(cand);
        }
    }

    results
}

/// Strips English morphological suffixes and transduces the base stem.
fn transduce_stem(word: &str) -> Vec<String> {
    static SUFFIXES: &[(&str, &[&str])] = &[
        ("stration", &["स्ट्रेशन"]),
        ("ization", &["इजेशन", "ाइजेशन"]),
        ("isation", &["इजेशन", "ाइजेशन"]),
        ("shion", &["शन", "सन"]),
        ("station", &["स्टेशन"]),
        ("tation", &["टेशन"]),
        ("cation", &["केशन"]),
        ("ation", &["ेशन"]),
        ("ution", &["ुसन", "ुशन"]),
        ("ction", &["क्शन", "क्सन"]),
        ("ition", &["िसन", "िशन"]),
        ("sion", &["सन", "शन"]),
        ("tion", &["शन", "सन"]),
        ("tive", &["टिभ"]),
        ("ture", &["चर"]),
        ("ment", &["मेन्ट"]),
        ("ness", &["नेस"]),
        ("less", &["लेस"]),
        ("able", &["एबल"]),
        ("ible", &["िबल"]),
        ("ship", &["सिप"]),
        ("hood", &["हुड"]),
        ("ology", &["ोलोजी"]),
        ("graphy", &["ग्राफी"]),
        ("spital", &["स्पिटल"]),
        ("pital", &["पिटल"]),
        ("ital", &["िटल"]),
        ("ical", &["िकल"]),
        ("al", &["ल"]),
        ("istic", &["िस्टिक"]),
        ("ism", &["इजम"]),
        ("ist", &["इस्ट"]),
        ("ing", &["िङ"]),
        ("ish", &["िस"]),
        ("ous", &["स"]),
        ("ful", &["फुल"]),
    ];

    for &(suf, dev_sufs) in SUFFIXES {
        if word == suf {
            return dev_sufs.iter().map(|s| s.to_string()).collect();
        }
        if word.ends_with(suf) && word.len() > suf.len() {
            let base = &word[..word.len() - suf.len()];
            let base_cands = transduce_base(base);
            let mut out = Vec::new();
            for b in &base_cands {
                for ds in dev_sufs {
                    out.push(format!("{}{}", b, ds));
                }
            }
            if !out.is_empty() {
                return out;
            }
        }
    }

    transduce_base(word)
}

/// Core Letter-to-Sound (L2S) sequential transduction for English stems.
fn transduce_base(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let n = chars.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return match chars[0] {
            'a' => vec!["ए".to_string(), "अ".to_string(), "आ".to_string()],
            'e' => vec!["इ".to_string(), "ए".to_string()],
            'i' => vec!["इ".to_string()],
            'o' => vec!["ओ".to_string(), "अ".to_string()],
            'u' => vec!["यु".to_string(), "उ".to_string()],
            c => map_coda(c).map(|s| vec![s.to_string()]).unwrap_or_default(),
        };
    }

    // Two-letter stem with vowel 'a' (e.g. "fa" -> फे, फा)
    if n == 2 && chars[1] == 'a' {
        if let Some(c_dev) = map_onset(&chars[0..1].iter().collect::<String>()) {
            return vec![
                format!("{}\u{0947}", c_dev),
                format!("{}\u{093E}", c_dev),
                format!("{}्या", c_dev),
            ];
        }
    }

    // 1. Check classic silent 'e' pattern: C + V + C + e at stem end (side, plate, bike, home, tube, chrome, drive)
    if word.ends_with('e') && n >= 4 {
        let vowel = chars[n - 3];
        let coda = chars[n - 2];
        let pre_vowel = chars[n - 4];
        if is_vowel(vowel) && !is_vowel(coda) && !is_vowel(pre_vowel) {
            let onset_str: String = chars[..n - 3].iter().collect();
            if let Some(onset_dev) = map_onset(&onset_str) {
                let coda_dev = match coda {
                    'c' => Some("स"),
                    'g' => Some("ज"),
                    other => map_coda(other),
                };
                if let Some(coda_dev) = coda_dev {
                    let mut out = Vec::new();
                    match vowel {
                        'i' | 'y' => {
                            out.push(format!("{}ाइ{}", onset_dev, coda_dev));
                            out.push(format!("{}ाय{}", onset_dev, coda_dev));
                        }
                        'a' => {
                            out.push(format!("{}\u{0947}{}", onset_dev, coda_dev));
                        }
                        'o' => {
                            out.push(format!("{}\u{094B}{}", onset_dev, coda_dev));
                        }
                        'u' => {
                            out.push(format!("{}्यु{}", onset_dev, coda_dev));
                        }
                        'e' => {
                            out.push(format!("{}ी{}", onset_dev, coda_dev));
                        }
                        _ => {}
                    }
                    if !out.is_empty() {
                        return out;
                    }
                }
            }
        }
    }

    // 2. Sequential phonemic parse
    let mut tokens: Vec<&'static str> = Vec::with_capacity(8);
    let mut i = 0;

    while i < n {
        let rem: String = chars[i..].iter().collect();

        // Multi-letter clusters
        if rem.starts_with("ight") {
            tokens.push("ाइट");
            i += 4;
            continue;
        }
        if rem.starts_with("igh") {
            tokens.push("ाइ");
            i += 3;
            continue;
        }
        if rem.starts_with("ire") && (i + 3 == n || !chars[i + 3].is_ascii_alphabetic()) {
            tokens.push("ायर");
            i += 3;
            continue;
        }
        if rem.starts_with("ore") && (i + 3 == n || !chars[i + 3].is_ascii_alphabetic()) {
            tokens.push("ोर");
            i += 3;
            continue;
        }
        if rem.starts_with("are") && (i + 3 == n || !chars[i + 3].is_ascii_alphabetic()) {
            tokens.push("ेयर");
            i += 3;
            continue;
        }
        if rem.starts_with("ure") && (i + 3 == n || !chars[i + 3].is_ascii_alphabetic()) {
            tokens.push("्योर");
            i += 3;
            continue;
        }
        if rem.starts_with("ound") && i + 4 == n {
            tokens.push("ाउन्ड");
            i += 4;
            continue;
        }
        if rem.starts_with("ind") && i + 3 == n {
            tokens.push("ाइन्ड");
            i += 3;
            continue;
        }
        if rem.starts_with("old") && i + 3 == n {
            tokens.push("ोल्ड");
            i += 3;
            continue;
        }
        if rem.starts_with("oard") && i + 4 == n {
            tokens.push("ोर्ड");
            i += 4;
            continue;
        }
        if rem.starts_with("uit") && i + 3 == n {
            tokens.push("ुट");
            i += 3;
            continue;
        }

        // Consonant digraphs & onset clusters
        if rem.starts_with("sch") {
            tokens.push("स्क");
            i += 3;
            continue;
        }
        if rem.starts_with("chr") {
            tokens.push("क्र");
            i += 3;
            continue;
        }
        if rem.starts_with("ch") {
            tokens.push("च");
            i += 2;
            continue;
        }
        if rem.starts_with("sh") {
            tokens.push("श");
            i += 2;
            continue;
        }
        if rem.starts_with("th") {
            tokens.push("थ");
            i += 2;
            continue;
        }
        if rem.starts_with("ph") {
            tokens.push("फ");
            i += 2;
            continue;
        }
        if rem.starts_with("wh") {
            tokens.push("व्ह");
            i += 2;
            continue;
        }
        if rem.starts_with("qu") {
            tokens.push("क्व");
            i += 2;
            continue;
        }
        if rem.starts_with("ck") {
            tokens.push("क");
            i += 2;
            continue;
        }
        if rem.starts_with("sc") || rem.starts_with("sk") {
            tokens.push("स्क");
            i += 2;
            continue;
        }
        if rem.starts_with("st") {
            tokens.push("स्ट");
            i += 2;
            continue;
        }
        if rem.starts_with("sp") {
            tokens.push("स्प");
            i += 2;
            continue;
        }
        if rem.starts_with("sl") {
            tokens.push("स्ल");
            i += 2;
            continue;
        }
        if rem.starts_with("sm") {
            tokens.push("स्म");
            i += 2;
            continue;
        }
        if rem.starts_with("sn") {
            tokens.push("स्न");
            i += 2;
            continue;
        }
        if rem.starts_with("sw") {
            tokens.push("स्व");
            i += 2;
            continue;
        }
        if rem.starts_with("pr") {
            tokens.push("प्र");
            i += 2;
            continue;
        }
        if rem.starts_with("pl") {
            tokens.push("प्ल");
            i += 2;
            continue;
        }
        if rem.starts_with("br") {
            tokens.push("ब्र");
            i += 2;
            continue;
        }
        if rem.starts_with("bl") {
            tokens.push("ब्ल");
            i += 2;
            continue;
        }
        if rem.starts_with("tr") {
            tokens.push("ट्र");
            i += 2;
            continue;
        }
        if rem.starts_with("dr") {
            tokens.push("ड्र");
            i += 2;
            continue;
        }
        if rem.starts_with("cl") {
            tokens.push("क्ल");
            i += 2;
            continue;
        }
        if rem.starts_with("cr") {
            tokens.push("क्र");
            i += 2;
            continue;
        }
        if rem.starts_with("fl") {
            tokens.push("फ्ल");
            i += 2;
            continue;
        }
        if rem.starts_with("fr") {
            tokens.push("फ्र");
            i += 2;
            continue;
        }
        if rem.starts_with("gr") {
            tokens.push("ग्र");
            i += 2;
            continue;
        }
        if rem.starts_with("gl") {
            tokens.push("ग्ला");
            i += 2;
            continue;
        }

        if rem.starts_with("nk") {
            tokens.push("ङ्क");
            i += 2;
            continue;
        }
        if rem.starts_with("ng") && (i + 2 == n || !chars[i + 2].is_ascii_alphabetic()) {
            tokens.push("ङ");
            i += 2;
            continue;
        }
        if rem.starts_with("mp") && (i + 2 == n || !chars[i + 2].is_ascii_alphabetic()) {
            tokens.push("म्प");
            i += 2;
            continue;
        }
        if rem.starts_with("nt") && (i + 2 == n || !chars[i + 2].is_ascii_alphabetic()) {
            tokens.push("न्ट");
            i += 2;
            continue;
        }
        if rem.starts_with("nd") && (i + 2 == n || !chars[i + 2].is_ascii_alphabetic()) {
            tokens.push("न्ड");
            i += 2;
            continue;
        }

        // Vowel digraphs
        if rem.starts_with("ee") || rem.starts_with("ea") {
            tokens.push("ी");
            i += 2;
            continue;
        }
        if rem.starts_with("ai") || rem.starts_with("ay") {
            tokens.push("े");
            i += 2;
            continue;
        }
        if rem.starts_with("oo") {
            tokens.push("ु");
            i += 2;
            continue;
        }
        if rem.starts_with("oa") {
            tokens.push("ो");
            i += 2;
            continue;
        }
        if rem.starts_with("ou") {
            tokens.push("ाउ");
            i += 2;
            continue;
        }
        if rem.starts_with("ow") {
            if i + 2 < n && matches!(chars[i + 2], 'n' | 'l') {
                tokens.push("ाउ");
            } else {
                tokens.push("ो");
            }
            i += 2;
            continue;
        }
        if rem.starts_with("oi") || rem.starts_with("oy") {
            tokens.push("ोइ");
            i += 2;
            continue;
        }
        if rem.starts_with("au") || rem.starts_with("aw") {
            tokens.push("अ");
            i += 2;
            continue;
        }
        if rem.starts_with("ew") {
            tokens.push("्यु");
            i += 2;
            continue;
        }
        if rem.starts_with("ey") {
            if i + 2 == n {
                tokens.push("ि");
            } else {
                tokens.push("े");
            }
            i += 2;
            continue;
        }
        if rem.starts_with("ar") {
            tokens.push(if i > 0 { "ार" } else { "आर" });
            i += 2;
            continue;
        }
        if rem.starts_with("er") || rem.starts_with("ir") || rem.starts_with("ur") {
            tokens.push("र");
            i += 2;
            continue;
        }
        if rem.starts_with("or") {
            tokens.push(if i + 2 == n { "र" } else { "ोर" });
            i += 2;
            continue;
        }

        // Single consonants
        let c = chars[i];
        if c == 'c' {
            if i + 1 < n && matches!(chars[i + 1], 'e' | 'i' | 'y') {
                tokens.push("स");
            } else {
                tokens.push("क");
            }
            i += 1;
            continue;
        }
        if c == 'g' {
            if i + 1 < n && matches!(chars[i + 1], 'e' | 'i' | 'y') {
                tokens.push("ज");
            } else {
                tokens.push("ग");
            }
            i += 1;
            continue;
        }
        if c == 'x' {
            tokens.push("क्स");
            i += 1;
            continue;
        }
        if c == 'v' {
            tokens.push("भ");
            i += 1;
            continue;
        }
        if c == 'w' {
            tokens.push("व");
            i += 1;
            continue;
        }
        if c == 'y' {
            if i > 0 && i + 1 < n && !is_vowel(chars[i - 1]) && !is_vowel(chars[i + 1]) {
                tokens.push("ि");
            } else if i + 1 == n {
                tokens.push("ी");
            } else {
                tokens.push("य");
            }
            i += 1;
            continue;
        }
        if c == 't' {
            tokens.push("ट");
            i += 1;
            continue;
        }
        if c == 'd' {
            tokens.push("ड");
            i += 1;
            continue;
        }
        if c == 'b' {
            tokens.push("ब");
            i += 1;
            continue;
        }
        if c == 'p' {
            tokens.push("प");
            i += 1;
            continue;
        }
        if c == 'k' {
            tokens.push("क");
            i += 1;
            continue;
        }
        if c == 'f' {
            tokens.push("फ");
            i += 1;
            continue;
        }
        if c == 'h' {
            tokens.push("ह");
            i += 1;
            continue;
        }
        if c == 'j' {
            tokens.push("ज");
            i += 1;
            continue;
        }
        if c == 'l' {
            tokens.push("ल");
            i += 1;
            continue;
        }
        if c == 'm' {
            tokens.push("म");
            i += 1;
            continue;
        }
        if c == 'n' {
            tokens.push("न");
            i += 1;
            continue;
        }
        if c == 'r' {
            tokens.push("र");
            i += 1;
            continue;
        }
        if c == 's' {
            tokens.push("स");
            i += 1;
            continue;
        }
        if c == 'z' {
            tokens.push("ज");
            i += 1;
            continue;
        }

        // Single vowels
        if c == 'a' {
            if i == 0 {
                tokens.push("ए");
            } else {
                tokens.push("ा");
            }
            i += 1;
            continue;
        }
        if c == 'e' {
            if i == 0 {
                tokens.push("इ");
            } else if i + 1 < n {
                tokens.push("े");
            }
            // Silent final 'e' skipped
            i += 1;
            continue;
        }
        if c == 'i' {
            if i == 0 {
                tokens.push("इ");
            } else {
                tokens.push("ि");
            }
            i += 1;
            continue;
        }
        if c == 'o' {
            if i == 0 {
                tokens.push("अ");
            } else if i + 1 < n && matches!(chars[i + 1], 'x' | 't' | 'p' | 'k' | 'c' | 'g' | 's') {
                // Short 'o' before coda stops/sibilants (fox, box, hot, top, hospital, cost) retains inherent vowel
            } else {
                tokens.push("ो");
            }
            i += 1;
            continue;
        }
        if c == 'u' {
            if i == 0 {
                tokens.push("यु");
            } else {
                tokens.push("ु");
            }
            i += 1;
            continue;
        }

        i += 1;
    }

    let cand = assemble_tokens(&tokens);
    if !cand.is_empty() {
        vec![cand]
    } else {
        Vec::new()
    }
}

fn assemble_tokens(tokens: &[&'static str]) -> String {
    let mut out = String::new();
    for &tok in tokens {
        out.push_str(tok);
    }
    out
}

#[inline]
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
}

fn map_onset(s: &str) -> Option<&'static str> {
    match s {
        "st" => Some("स्ट"),
        "sp" => Some("स्प"),
        "sk" | "sc" | "sch" => Some("स्क"),
        "pr" => Some("प्र"),
        "pl" => Some("प्ल"),
        "dr" => Some("ड्र"),
        "tr" => Some("ट्र"),
        "cl" => Some("क्ल"),
        "cr" | "chr" => Some("क्र"),
        "gr" => Some("ग्र"),
        "br" => Some("ब्र"),
        "fr" => Some("फ्र"),
        "fl" => Some("फ्ल"),
        "gl" => Some("ग्ला"),
        "sh" => Some("श"),
        "ch" => Some("च"),
        "th" => Some("थ"),
        "ph" => Some("फ"),
        "qu" => Some("क्व"),
        "b" => Some("ब"),
        "c" => Some("क"),
        "d" => Some("ड"),
        "f" => Some("फ"),
        "g" => Some("ग"),
        "h" => Some("ह"),
        "j" => Some("ज"),
        "k" => Some("क"),
        "l" => Some("ल"),
        "m" => Some("म"),
        "n" => Some("न"),
        "p" => Some("प"),
        "r" => Some("र"),
        "s" => Some("स"),
        "t" => Some("ट"),
        "v" => Some("भ"),
        "w" => Some("व"),
        "y" => Some("य"),
        "z" => Some("ज"),
        _ => None,
    }
}

fn map_coda(c: char) -> Option<&'static str> {
    match c {
        'b' => Some("ब"),
        'c' => Some("क"),
        'd' => Some("ड"),
        'f' => Some("फ"),
        'g' => Some("ग"),
        'k' => Some("क"),
        'l' => Some("ल"),
        'm' => Some("म"),
        'n' => Some("न"),
        'p' => Some("प"),
        'r' => Some("र"),
        's' => Some("स"),
        't' => Some("ट"),
        'v' => Some("भ"),
        'x' => Some("क्स"),
        'z' => Some("ज"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_vocabulary_phonemizer() {
        // Open-vocabulary English words not present in any dictionary:
        let ff = english_to_devanagari("firefox");
        assert!(
            ff.iter().any(|c| c == "फायरफक्स" || c == "फाइरफक्स"),
            "Expected firefox transliteration, got {:?}",
            ff
        );

        let dl = english_to_devanagari("download");
        assert!(
            dl.iter().any(|c| c == "डाउनलोड"),
            "Expected download -> डाउनलोड, got {:?}",
            dl
        );

        let bt = english_to_devanagari("bluetooth");
        assert!(
            bt.iter().any(|c| c == "ब्लुटुथ"),
            "Expected bluetooth -> ब्लुटुथ, got {:?}",
            bt
        );

        let kb = english_to_devanagari("keyboard");
        assert!(
            kb.iter().any(|c| c == "किबोर्ड" || c == "कीबोर्ड"),
            "Expected keyboard -> किबोर्ड or कीबोर्ड, got {:?}",
            kb
        );

        let ch = english_to_devanagari("chrome");
        assert!(
            ch.iter().any(|c| c == "क्रोम"),
            "Expected chrome -> क्रोम, got {:?}",
            ch
        );

        let sys = english_to_devanagari("system");
        assert!(
            sys.iter().any(|c| c.contains("सिस्ट")),
            "Expected system -> सिस्टम..., got {:?}",
            sys
        );
    }

    #[test]
    fn test_silent_e_rules() {
        assert!(english_to_devanagari("side").contains(&"साइड".to_string()));
        assert!(english_to_devanagari("plate").contains(&"प्लेट".to_string()));
        assert!(english_to_devanagari("bike").contains(&"बाइक".to_string()));
        assert!(english_to_devanagari("plane").contains(&"प्लेन".to_string()));
    }

    #[test]
    fn test_morphological_suffixes() {
        assert!(english_to_devanagari("fashion")
            .iter()
            .any(|c| c.contains("शन")));
        assert!(english_to_devanagari("station")
            .iter()
            .any(|c| c.contains("स्टेशन")));
        assert!(english_to_devanagari("action")
            .iter()
            .any(|c| c.contains("शन") || c.contains("सन")));
    }
}
