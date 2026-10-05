// File: src/core/english_phonetic.rs
//
// Lightweight, zero-dependency English-to-Devanagari phonetic transducer.
//
// Converts English loanwords and Latin orthography into standard Nepali Devanagari
// using rule-based grapheme-to-phoneme (G2P) and common loanword mappings.

/// Transduces an English word into possible Devanagari transliterations.
///
/// Returns high-confidence phonetic candidates. Returns empty Vec if the word
/// does not match English loanword patterns.
pub fn english_to_devanagari(word: &str) -> Vec<String> {
    let lower = word.to_ascii_lowercase();
    let lower = lower.trim();
    if lower.len() < 2 || !lower.chars().all(|c| c.is_ascii_alphabetic()) {
        return Vec::new();
    }

    let mut results = Vec::with_capacity(3);

    // 1. Exact high-frequency English loanword dictionary (Nepali conventional spellings)
    if let Ok(idx) = COMMON_ENGLISH_LOANS.binary_search_by_key(&lower, |&(k, _)| k) {
        results.push(COMMON_ENGLISH_LOANS[idx].1.to_string());
    }

    // 2. Rule-based Letter-to-Sound (L2S) transduction
    if let Some(dev) = rule_transduce(lower) {
        if !results.contains(&dev) {
            results.push(dev);
        }
    }

    results
}

/// Common English loanwords in Nepali with conventional Devanagari orthography.
/// Kept sorted alphabetically for O(log N) binary search.
static COMMON_ENGLISH_LOANS: &[(&str, &str)] = &[
    ("action", "एक्शन"),
    ("administration", "एडमिनिस्ट्रेशन"),
    ("airport", "एयरपोर्ट"),
    ("apple", "स्याउ"),
    ("bag", "ब्याग"),
    ("bank", "बैंक"),
    ("biscuit", "बिस्कुट"),
    ("board", "बोर्ड"),
    ("book", "बुक"),
    ("bus", "बस"),
    ("camp", "क्याम्प"),
    ("captain", "कप्तान"),
    ("card", "कार्ड"),
    ("car", "कार"),
    ("case", "केस"),
    ("cat", "क्याट"),
    ("chair", "चेयर"),
    ("chance", "चान्स"),
    ("check", "चेक"),
    ("city", "सिटी"),
    ("class", "क्लास"),
    ("club", "क्लब"),
    ("coach", "कोच"),
    ("code", "कोड"),
    ("college", "कलेज"),
    ("committee", "कमिटी"),
    ("company", "कम्पनी"),
    ("computer", "कम्प्युटर"),
    ("consultative", "कन्सलटेटिभ"),
    ("court", "कोर्ट"),
    ("cup", "कप"),
    ("date", "डेट"),
    ("day", "डे"),
    ("department", "डिपार्टमेन्ट"),
    ("desk", "डेस्क"),
    ("diary", "डायरी"),
    ("doctor", "डाक्टर"),
    ("driver", "ड्राइभर"),
    ("face", "फेस"),
    ("factory", "फ्याक्ट्री"),
    ("fan", "फ्यान"),
    ("fashion", "फेशन"),
    ("file", "फाइल"),
    ("film", "फिल्म"),
    ("food", "फुड"),
    ("game", "गेम"),
    ("gas", "ग्यास"),
    ("gate", "गेट"),
    ("general", "जनरल"),
    ("glass", "ग्लास"),
    ("goal", "गोल"),
    ("grade", "ग्रेड"),
    ("group", "ग्रुप"),
    ("hall", "हल"),
    ("head", "हेड"),
    ("home", "होम"),
    ("hospital", "हस्पिटल"),
    ("hotel", "होटेल"),
    ("internet", "इन्टरनेट"),
    ("jeep", "जिप"),
    ("judge", "जज"),
    ("late", "लेट"),
    ("leader", "लिडर"),
    ("line", "लाइन"),
    ("list", "लिस्ट"),
    ("lock", "लक"),
    ("map", "म्याप"),
    ("match", "म्याच"),
    ("media", "मिडिया"),
    ("meet", "मिट"),
    ("meeting", "मिटिङ"),
    ("member", "मेम्बर"),
    ("mobile", "मोबाइल"),
    ("model", "मोडल"),
    ("net", "नेट"),
    ("news", "न्यूज"),
    ("note", "नोट"),
    ("number", "नम्बर"),
    ("office", "अफिस"),
    ("online", "अनलाइन"),
    ("pass", "पास"),
    ("pen", "पेन"),
    ("phone", "फोन"),
    ("photo", "फोटो"),
    ("picture", "पिक्चर"),
    ("plan", "प्लान"),
    ("plantation", "प्लान्टेशन"),
    ("plate", "प्लेट"),
    ("point", "पोइन्ट"),
    ("police", "पुलिस"),
    ("post", "पोस्ट"),
    ("preliminary", "प्रिलिमिनेरी"),
    ("press", "प्रेस"),
    ("problem", "प्रब्लम"),
    ("professor", "प्रोफेसर"),
    ("project", "प्रोजेक्ट"),
    ("public", "पब्लिक"),
    ("radio", "रेडियो"),
    ("rail", "रेल"),
    ("rate", "रेट"),
    ("record", "रेकर्ड"),
    ("report", "रिपोर्ट"),
    ("road", "रोड"),
    ("room", "रुम"),
    ("round", "राउन्ड"),
    ("school", "स्कुल"),
    ("scottish", "स्कटिस"),
    ("seat", "सिट"),
    ("service", "सर्विस"),
    ("shirt", "सर्ट"),
    ("side", "साइड"),
    ("size", "साइज"),
    ("sound", "साउन्ड"),
    ("speed", "स्पिड"),
    ("sport", "स्पोर्ट"),
    ("staff", "स्टाफ"),
    ("stage", "स्टेज"),
    ("station", "स्टेशन"),
    ("stop", "स्टप"),
    ("switzerland", "स्वीटजरल्याण्ड"),
    ("system", "सिस्टम"),
    ("table", "टेबल"),
    ("team", "टिम"),
    ("test", "टेस्ट"),
    ("ticket", "टिकट"),
    ("time", "टाइम"),
    ("tour", "टुर"),
    ("track", "ट्र्याक"),
    ("train", "ट्रेन"),
    ("truck", "ट्रक"),
    ("van", "भ्यान"),
    ("video", "भिडियो"),
    ("view", "भ्यू"),
    ("vote", "भोट"),
    ("ward", "वार्ड"),
    ("zone", "जोन"),
];

/// Algorithmic Letter-to-Sound (L2S) rule transduction for English loanwords.
fn rule_transduce(word: &str) -> Option<String> {
    if word.len() < 3 {
        return None;
    }

    // 1. Suffix stripping
    let suffixes: &[(&str, &str)] = &[
        ("tion", "शन"),
        ("sion", "शन"),
        ("tive", "टिभ"),
        ("ture", "चर"),
        ("ment", "मेन्ट"),
        ("ing", "िङ"),
        ("able", "एबल"),
        ("ness", "नेस"),
        ("less", "लेस"),
    ];

    let mut stem = word;
    let mut dev_suffix = "";
    for &(suf, dev_suf) in suffixes {
        if word.ends_with(suf) && word.len() > suf.len() + 1 {
            stem = &word[..word.len() - suf.len()];
            dev_suffix = dev_suf;
            break;
        }
    }

    // 2. Silent "e" vowel lengthening pattern: C + V + C + e
    if stem.ends_with('e') && stem.len() >= 4 {
        let chars: Vec<char> = stem.chars().collect();
        let n = chars.len();
        let vowel = chars[n - 3];
        let coda = chars[n - 2];
        let pre_vowel = chars[n - 4];

        if is_vowel(vowel) && !is_vowel(coda) && !is_vowel(pre_vowel) {
            let onset_str: String = chars[..n - 3].iter().collect();
            let onset_dev = map_consonants(&onset_str)?;
            let coda_dev = map_single_consonant(coda)?;

            let out = match vowel {
                'i' | 'y' => format!("{}ाइ{}{}", onset_dev, coda_dev, dev_suffix),
                'a' => format!("{}\u{0947}{}{}", onset_dev, coda_dev, dev_suffix),
                'o' => format!("{}\u{094B}{}{}", onset_dev, coda_dev, dev_suffix),
                'u' => format!("{}्यु{}{}", onset_dev, coda_dev, dev_suffix),
                _ => return None,
            };
            return Some(out);
        }
    }

    None
}

#[inline]
fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u')
}

fn map_single_consonant(c: char) -> Option<&'static str> {
    match c {
        'b' => Some("ब"),
        'c' => Some("क"),
        'd' => Some("ड"),
        'f' => Some("फ"),
        'g' => Some("ग"),
        'h' => Some("ह"),
        'j' => Some("ज"),
        'k' => Some("क"),
        'l' => Some("ल"),
        'm' => Some("म"),
        'n' => Some("न"),
        'p' => Some("प"),
        'r' => Some("र"),
        's' => Some("स"),
        't' => Some("ट"),
        'v' => Some("भ"),
        'w' => Some("व"),
        'x' => Some("क्स"),
        'y' => Some("य"),
        'z' => Some("ज"),
        _ => None,
    }
}

fn map_consonants(s: &str) -> Option<String> {
    match s {
        "st" => Some("स्ट".to_string()),
        "sp" => Some("स्प".to_string()),
        "sk" | "sc" => Some("स्क".to_string()),
        "sch" => Some("स्क".to_string()),
        "pr" => Some("प्र".to_string()),
        "pl" => Some("प्ल".to_string()),
        "dr" => Some("ड्र".to_string()),
        "tr" => Some("ट्र".to_string()),
        "cl" => Some("क्ल".to_string()),
        "cr" => Some("क्र".to_string()),
        "gr" => Some("ग्र".to_string()),
        "br" => Some("ब्र".to_string()),
        "fr" => Some("फ्र".to_string()),
        "fl" => Some("फ्ल".to_string()),
        "gl" => Some("ग्ला".to_string()),
        "sh" => Some("श".to_string()),
        "ch" => Some("च".to_string()),
        "th" => Some("थ".to_string()),
        "ph" => Some("फ".to_string()),
        "qu" => Some("क्व".to_string()),
        _ if s.len() == 1 => map_single_consonant(s.chars().next()?).map(ToString::to_string),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transduces_exact_common_loans() {
        assert_eq!(english_to_devanagari("side"), vec!["साइड"]);
        assert_eq!(english_to_devanagari("plate"), vec!["प्लेट"]);
        assert_eq!(english_to_devanagari("fashion"), vec!["फेशन"]);
        assert_eq!(english_to_devanagari("school"), vec!["स्कुल"]);
        assert_eq!(english_to_devanagari("food"), vec!["फुड"]);
        assert_eq!(english_to_devanagari("action"), vec!["एक्शन"]);
        assert_eq!(english_to_devanagari("biscuit"), vec!["बिस्कुट"]);
        assert_eq!(english_to_devanagari("public"), vec!["पब्लिक"]);
        assert_eq!(english_to_devanagari("bag"), vec!["ब्याग"]);
        assert_eq!(english_to_devanagari("administration"), vec!["एडमिनिस्ट्रेशन"]);
        assert_eq!(english_to_devanagari("consultative"), vec!["कन्सलटेटिभ"]);
    }

    #[test]
    fn transduces_silent_e_rules() {
        // Words not in the common dictionary, handled purely by rule:
        assert_eq!(english_to_devanagari("bike"), vec!["बाइक"]);
        assert_eq!(english_to_devanagari("hike"), vec!["हाइक"]);
        assert_eq!(english_to_devanagari("lake"), vec!["लेक"]);
    }
}
