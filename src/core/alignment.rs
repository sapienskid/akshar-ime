// File: src/core/alignment.rs
//
// Character-level alignment between a Roman string and a Devanagari string.
//
// A plain Levenshtein DP only permits 1:1 substitutions, insertions and
// deletions. Brahmic transliteration frequently needs n:1 mappings where a
// *bigram* of Roman characters maps to a single Devanagari codepoint, e.g.:
//
//     "kh" -> ख      "aa" -> ा      "ng" -> ङ
//
// To capture these we extend the edit graph with a fourth operation, MERGE,
// that consumes two Roman characters and emits one Devanagari codepoint. The
// alignment cost model is:
//
//     SUB   1.0   (one Roman char  <-> one Devanagari char)
//     DEL   1.5   (drop a Roman char with no Devanagari output)
//     INS   1.5   (emit a Devanagari char with no Roman input)
//     MERGE 1.3   (two Roman chars -> one Devanagari char)
//
// SUB is cheaper than MERGE, so 1:1 alignments are preferred when the two
// strings have equal length; MERGE is cheaper than SUB+DEL, so digraphs win
// over "substitute then delete" in length-mismatched regions. The back-trace
// returns, for each Devanagari codepoint, the Roman substring (length 0, 1 or
// 2) that was aligned to it. Callers drop the empty (insertion) pairs when
// building emission tables.

/// Operation codes used in the DP back-trace.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Op {
    Sub = 0,
    DelRoman = 1,
    InsDev = 2,
    Merge = 3,
}

const SUB: f64 = 1.0;
const DEL: f64 = 1.5;
const INS: f64 = 1.5;
const MERGE: f64 = 1.3;

/// One aligned pair: the Roman substring (0, 1 or 2 chars) and the Devanagari
/// codepoint it was aligned to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignedPair {
    pub roman: String,
    pub dev_char: char,
}

/// Align `roman` to `dev` at the character level, returning one `AlignedPair`
/// per Devanagari codepoint (in left-to-right order). Pairs with an empty
/// `roman` field are insertions (a Devanagari codepoint with no Roman source).
pub fn align(roman: &str, dev: &str) -> Vec<AlignedPair> {
    let r: Vec<char> = roman.chars().collect();
    let d: Vec<char> = dev.chars().collect();
    let m = r.len();
    let n = d.len();

    if n == 0 {
        return Vec::new();
    }
    if m == 0 {
        // Every Devanagari char is an insertion.
        return d.iter().map(|&c| AlignedPair { roman: String::new(), dev_char: c }).collect();
    }

    // dp[i][j] = minimum cost to align r[..i] with d[..j].
    let mut dp = vec![vec![f64::INFINITY; n + 1]; m + 1];
    let mut bp = vec![vec![Op::Sub; n + 1]; m + 1];

    dp[0][0] = 0.0;
    for i in 1..=m {
        dp[i][0] = dp[i - 1][0] + DEL;
        bp[i][0] = Op::DelRoman;
    }
    for j in 1..=n {
        dp[0][j] = dp[0][j - 1] + INS;
        bp[0][j] = Op::InsDev;
    }

    for i in 1..=m {
        for j in 1..=n {
            let sub = dp[i - 1][j - 1] + SUB;
            let del = dp[i - 1][j] + DEL;
            let ins = dp[i][j - 1] + INS;

            let mut best = sub;
            let mut op = Op::Sub;
            if del < best {
                best = del;
                op = Op::DelRoman;
            }
            if ins < best {
                best = ins;
                op = Op::InsDev;
            }
            if i >= 2 {
                let merge = dp[i - 2][j - 1] + MERGE;
                if merge < best {
                    best = merge;
                    op = Op::Merge;
                }
            }
            dp[i][j] = best;
            bp[i][j] = op;
        }
    }

    // Back-trace from (m, n) to (0, 0), collecting pairs in reverse.
    let mut pairs: Vec<AlignedPair> = Vec::new();
    let (mut i, mut j) = (m, n);
    while i > 0 || j > 0 {
        match bp[i][j] {
            Op::Sub => {
                pairs.push(AlignedPair { roman: r[i - 1].to_string(), dev_char: d[j - 1] });
                i -= 1;
                j -= 1;
            }
            Op::DelRoman => {
                // Roman char with no Devanagari counterpart: emit nothing.
                i -= 1;
            }
            Op::InsDev => {
                pairs.push(AlignedPair { roman: String::new(), dev_char: d[j - 1] });
                j -= 1;
            }
            Op::Merge => {
                let bigram: String = r[i - 2..i].iter().collect();
                pairs.push(AlignedPair { roman: bigram, dev_char: d[j - 1] });
                i -= 2;
                j -= 1;
            }
        }
    }
    pairs.reverse();
    pairs
}

/// Convenience: return only the aligned pairs that have a non-empty Roman
/// source (i.e. drop pure insertions). These are the pairs used to build
/// the emission table.
pub fn align_emissive(roman: &str, dev: &str) -> Vec<AlignedPair> {
    align(roman, dev)
        .into_iter()
        .filter(|p| !p.roman.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligns_ram_to_rama() {
        // "ram" -> "रम" (र, म). roman has 3 chars, dev has 2, so the DP merges
        // the inherent-schwa 'a' into the first akshara. All roman chars must be
        // consumed and every Devanagari char must get a non-empty source.
        let pairs = align_emissive("ram", "रम");
        let dev_seq: String = pairs.iter().map(|p| p.dev_char).collect();
        let roman_seq: String = pairs.iter().map(|p| p.roman.as_str()).collect();
        assert_eq!(dev_seq, "रम");
        assert_eq!(roman_seq, "ram");
    }

    #[test]
    fn aligns_digraph_via_merge() {
        // "kh" -> "ख": should merge the bigram.
        let pairs = align_emissive("kh", "ख");
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].roman, "kh");
        assert_eq!(pairs[0].dev_char, 'ख');
    }

    #[test]
    fn merge_preferred_over_sub_plus_delete() {
        // "ka" -> "क" (1 Devanagari char). Merge "ka"->क should beat sub+del.
        let pairs = align_emissive("ka", "क");
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].roman, "ka");
        assert_eq!(pairs[0].dev_char, 'क');
    }

    #[test]
    fn aligns_matra_to_vowel_char() {
        // "kaa" -> "का" = क + ा. roman "kaa" (3) vs dev 2 chars.
        // Expected: "ka"->क (merge) and "a"->ा (sub), OR "k"->क and "aa"->ा.
        // Both are acceptable; just check क and ा both get a non-empty roman.
        let pairs = align_emissive("kaa", "का");
        let chars: Vec<char> = pairs.iter().map(|p| p.dev_char).collect();
        assert!(chars.contains(&'क'));
        assert!(chars.contains(&'ा'));
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn empty_roman_yields_all_insertions() {
        let pairs = align("", "कम");
        assert_eq!(pairs.len(), 2);
        assert!(pairs.iter().all(|p| p.roman.is_empty()));
    }

    #[test]
    fn empty_devanagari_yields_nothing() {
        let pairs = align("abc", "");
        assert!(pairs.is_empty());
    }

    #[test]
    fn align_emissive_drops_insertions() {
        // "k" -> "कम": one Devanagari char has no roman source (insertion).
        let full = align("k", "कम");
        assert_eq!(full.len(), 2);
        let emissive = align_emissive("k", "कम");
        assert!(emissive.len() < full.len());
        assert!(emissive.iter().all(|p| !p.roman.is_empty()));
    }

    #[test]
    fn aligns_longer_word_namaste() {
        let pairs = align_emissive("namaste", "नमस्ते");
        // Every Devanagari char (न म स त े) should get a non-empty roman source.
        let chars: String = pairs.iter().map(|p| p.dev_char).collect();
        assert_eq!(chars, "नमस्ते");
        // Roman sources should reconstruct to length 7 (all chars consumed).
        let total: usize = pairs.iter().map(|p| p.roman.chars().count()).sum();
        assert_eq!(total, 7);
    }

    #[test]
    fn aligned_pairs_are_in_devanagari_order() {
        let pairs = align_emissive("namaste", "नमस्ते");
        let dev_seq: String = pairs.iter().map(|p| p.dev_char).collect();
        assert_eq!(dev_seq, "नमस्ते");
    }
}
