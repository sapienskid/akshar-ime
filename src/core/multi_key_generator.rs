//! 4-Stream Multi-Key Phonetic Projection & Pairwise Rank #1/#2 Matra Disambiguator.

pub fn project_multi_keys(devanagari: &str) -> Vec<String> {
    let chars: Vec<char> = devanagari.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }

    let mut canonical = String::new();
    let mut schwa_dropped = String::new();
    let mut colloquial = String::new();
    let mut skeleton = String::new();

    let n = chars.len();
    for i in 0..n {
        let c = chars[i];
        let next = if i + 1 < n { Some(chars[i + 1]) } else { None };
        let is_word_final = i + 1 == n;

        if let Some((rom_std, rom_colloq)) = map_consonant(c) {
            canonical.push_str(rom_std);
            schwa_dropped.push_str(rom_std);
            colloquial.push_str(rom_colloq);
            skeleton.push_str(rom_std);

            let has_explicit_matra_or_halanta =
                next.is_some_and(|nc| is_matra(nc) || nc == '\u{094D}');

            if !has_explicit_matra_or_halanta {
                canonical.push('a');
                colloquial.push('a');
                if i == 0 || is_word_final {
                    schwa_dropped.push('a');
                }
            }
        } else if let Some(v_rom) = map_independent_vowel_or_matra(c) {
            canonical.push_str(v_rom);
            schwa_dropped.push_str(v_rom);
            colloquial.push_str(v_rom);
            if i == 0 && !is_matra(c) {
                skeleton.push_str(v_rom);
            }
        } else if c == 'ं' || c == 'ँ' {
            canonical.push('n');
            schwa_dropped.push('n');
            colloquial.push('n');
            skeleton.push('n');
        }
    }

    let mut keys = Vec::with_capacity(4);
    for k in [canonical, schwa_dropped, colloquial, skeleton] {
        if !k.is_empty() && !keys.contains(&k) {
            keys.push(k);
        }
    }
    keys
}

/// Project only phonetic and colloquial streams (canonical, schwa-dropped, colloquial).
/// Explicitly excludes the unvoweled `skeleton` stream so consonant clusters (e.g. `pm`, `km`)
/// never falsely promote arbitrary Sanskrit/Devanagari consonant skeletons in tie-breaking.
pub fn project_colloquial_keys(devanagari: &str) -> Vec<String> {
    let chars: Vec<char> = devanagari.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }

    let mut canonical = String::new();
    let mut schwa_dropped = String::new();
    let mut colloquial = String::new();

    let n = chars.len();
    for i in 0..n {
        let c = chars[i];
        let next = if i + 1 < n { Some(chars[i + 1]) } else { None };
        let is_word_final = i + 1 == n;

        if let Some((rom_std, rom_colloq)) = map_consonant(c) {
            canonical.push_str(rom_std);
            schwa_dropped.push_str(rom_std);
            colloquial.push_str(rom_colloq);

            let has_explicit_matra_or_halanta =
                next.is_some_and(|nc| is_matra(nc) || nc == '\u{094D}');

            if !has_explicit_matra_or_halanta {
                canonical.push('a');
                colloquial.push('a');
                if i == 0 || is_word_final {
                    schwa_dropped.push('a');
                }
            }
        } else if let Some(v_rom) = map_independent_vowel_or_matra(c) {
            canonical.push_str(v_rom);
            schwa_dropped.push_str(v_rom);
            colloquial.push_str(v_rom);
        } else if c == 'ं' || c == 'ँ' {
            canonical.push('n');
            schwa_dropped.push('n');
            colloquial.push('n');
        }
    }

    let mut keys = Vec::with_capacity(3);
    for k in [canonical, schwa_dropped, colloquial] {
        if !k.is_empty() && !keys.contains(&k) {
            keys.push(k);
        }
    }
    keys
}

fn is_matra(c: char) -> bool {
    matches!(c, 'ा' | 'ि' | 'ी' | 'ु' | 'ू' | 'ृ' | 'े' | 'ै' | 'ो' | 'ौ')
}

fn map_consonant(c: char) -> Option<(&'static str, &'static str)> {
    Some(match c {
        'क' => ("k", "k"),
        'ख' => ("kh", "kh"),
        'ग' => ("g", "g"),
        'घ' => ("gh", "gh"),
        'ङ' => ("ng", "ng"),
        'च' => ("ch", "ch"),
        'छ' => ("chh", "x"),
        'ज' => ("j", "j"),
        'झ' => ("jh", "jh"),
        'ञ' => ("ny", "n"),
        'ट' | 'त' => ("t", "t"),
        'ठ' | 'थ' => ("th", "th"),
        'ड' | 'द' => ("d", "d"),
        'ढ' | 'ध' => ("dh", "dh"),
        'ण' | 'न' => ("n", "n"),
        'प' => ("p", "p"),
        'फ' => ("ph", "f"),
        'ब' => ("b", "b"),
        'भ' => ("bh", "v"),
        'म' => ("m", "m"),
        'य' => ("y", "y"),
        'र' => ("r", "r"),
        'ल' => ("l", "l"),
        'व' => ("b", "w"),
        'श' | 'ष' | 'स' => ("s", "s"),
        'ह' => ("h", "h"),
        _ => return None,
    })
}

fn map_independent_vowel_or_matra(c: char) -> Option<&'static str> {
    Some(match c {
        'अ' | 'आ' | 'ा' => "a",
        'इ' | 'ई' | 'ि' | 'ी' => "i",
        'उ' | 'ऊ' | 'ु' | 'ू' => "u",
        'ऋ' | 'ृ' => "ri",
        'ए' | 'े' => "e",
        'ऐ' | 'ै' => "ai",
        'ओ' | 'ो' => "o",
        'औ' | 'ौ' => "au",
        _ => return None,
    })
}

pub fn consonant_matra_class_skeleton(devanagari: &str) -> String {
    let mut out = String::with_capacity(devanagari.len());
    for c in devanagari.chars() {
        match c {
            'ि' | 'ी' => out.push('I'),
            'ु' | 'ू' => out.push('U'),
            'इ' | 'ई' => out.push('i'),
            'उ' | 'ऊ' => out.push('u'),
            'श' | 'ष' | 'स' => out.push('स'),
            'ं' | 'ँ' => out.push('M'),
            _ => out.push(c),
        }
    }
    out
}

pub fn score_minimal_pair_matra(roman: &str, candidate: &str) -> f32 {
    let mut bonus = 0.0f32;
    let rom = roman.to_ascii_lowercase();

    let user_typed_long_i = rom.contains("ee") || rom.contains("ii");
    let user_typed_long_u = rom.contains("oo") || rom.contains("uu");
    let user_typed_long_a = rom.contains("aa");

    let has_dirgha_i = candidate.contains('ी') || candidate.contains('ई');
    let has_hrasva_i = candidate.contains('ि') || candidate.contains('इ');
    let has_dirgha_u = candidate.contains('ू') || candidate.contains('ऊ');
    let has_hrasva_u = candidate.contains('ु') || candidate.contains('उ');

    if user_typed_long_i {
        if has_dirgha_i {
            bonus += 1.25;
        } else if has_hrasva_i {
            bonus -= 0.75;
        }
    }
    if user_typed_long_u {
        if has_dirgha_u {
            bonus += 1.25;
        } else if has_hrasva_u {
            bonus -= 0.75;
        }
    }
    if user_typed_long_a && candidate.contains('ा') {
        bonus += 0.85;
    }

    if matches!(candidate, "पनि" | "कि" | "अनि" | "नि" | "तिरि")
        && !user_typed_long_i
        && !user_typed_long_a
    {
        bonus += 0.90;
    } else if candidate.ends_with('ी') && candidate.chars().count() >= 3 {
        bonus += 0.45;
    }

    let has_tatsama_marker = candidate.contains('ण')
        || candidate.contains('ष')
        || candidate.contains("क्ष")
        || candidate.contains("त्र")
        || candidate.contains("ज्ञ")
        || candidate.contains('ृ');
    if has_tatsama_marker
        && (candidate.contains('श') || candidate.contains('ष') || candidate.contains('व'))
    {
        let conflicts_s =
            (candidate.contains('श') || candidate.contains('ष')) && !rom.contains("sh");
        let conflicts_b = candidate.contains('व') && rom.starts_with('b');
        if !conflicts_s && !conflicts_b {
            bonus += 0.55;
        }
    }

    if rom.contains("sh") && (candidate.contains('श') || candidate.contains('ष')) {
        bonus += 0.80;
    }

    bonus
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_key_projections() {
        let keys_chha = project_multi_keys("छ");
        assert!(keys_chha.contains(&"xa".to_string()) || keys_chha.contains(&"chha".to_string()));

        let keys_bhayo = project_multi_keys("भयो");
        assert!(
            keys_bhayo.contains(&"vayo".to_string()) || keys_bhayo.contains(&"bhayo".to_string())
        );

        let keys_sathi = project_multi_keys("साथी");
        assert!(
            keys_sathi.contains(&"sathi".to_string()) || keys_sathi.contains(&"saathi".to_string())
        );
    }

    #[test]
    fn test_score_minimal_pair_matra() {
        // user typed 'saathee', 'साथी' should get bonus over 'साथि'
        let s_dirgha = score_minimal_pair_matra("saathee", "साथी");
        let s_hrasva = score_minimal_pair_matra("saathee", "साथि");
        assert!(s_dirgha > s_hrasva);

        // 'pani' without 'ee' or 'aa' should prioritize 'पनि'
        let p_pani = score_minimal_pair_matra("pani", "पनि");
        assert!(p_pani > 0.0);
    }
}
