// File: src/core/lexicon.rs
//
// Roman -> Devanagari word lexicon built from the training corpus.
//
// The Aksharantar Nepali corpus is ~2.4M (roman, devanagari) pairs with
// near-unique words, so it is a *vocabulary*, not a frequency distribution.
// We store the pairs sorted by roman for binary-search exact lookup and
// prefix scanning, and aggregate the distinct Devanagari spellings per roman
// prefix.  The IME layer combines lexicon evidence with the generative decoder
// and the user's own learning history.
//
// Storage is compact: roman and devanagari strings live in two byte arenas with
// per-entry offsets, so the runtime footprint is roughly the raw text size
// (~50 MB) rather than ~135 MB of boxed Strings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RomanLexicon {
    /// Lowercase ASCII roman strings, concatenated (length-prefixed per entry).
    romans: Vec<u8>,
    /// UTF-8 devanagari strings, concatenated (length-prefixed per entry).
    devs: Vec<u8>,
    /// Offset into `romans` for entry i (roman length in `roman_len[i]`).
    roman_off: Vec<u32>,
    roman_len: Vec<u8>,
    /// Offset into `devs` for entry i (dev length in `dev_len[i]`).
    dev_off: Vec<u32>,
    dev_len: Vec<u16>,
}

impl RomanLexicon {
    /// Build from (roman, devanagari) pairs; roman is lowercased.
    pub fn build<I>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (String, String)>,
    {
        let mut entries: Vec<(String, String)> = pairs
            .into_iter()
            .map(|(r, d)| (r.to_ascii_lowercase(), d))
            .filter(|(r, d)| {
                !r.is_empty()
                    && !d.is_empty()
                    && r.bytes().all(|b| b.is_ascii_lowercase())
                    && d.len() <= u16::MAX as usize
            })
            .collect();
        entries.sort();
        entries.dedup();

        let mut lx = Self::default();
        lx.romans.reserve(entries.iter().map(|(r, _)| r.len() + 1).sum());
        lx.devs.reserve(entries.iter().map(|(_, d)| d.len() + 1).sum());
        for (r, d) in entries {
            lx.roman_off.push(lx.romans.len() as u32);
            lx.roman_len.push(r.len() as u8);
            lx.romans.extend_from_slice(r.as_bytes());
            lx.dev_off.push(lx.devs.len() as u32);
            lx.dev_len.push(d.len() as u16);
            lx.devs.extend_from_slice(d.as_bytes());
        }
        lx
    }

    fn roman_at(&self, i: usize) -> &[u8] {
        let off = self.roman_off[i] as usize;
        let len = self.roman_len[i] as usize;
        &self.romans[off..off + len]
    }

    fn dev_at(&self, i: usize) -> &str {
        let off = self.dev_off[i] as usize;
        let len = self.dev_len[i] as usize;
        // Devanagari is always valid UTF-8 (built from String).
        std::str::from_utf8(&self.devs[off..off + len]).unwrap_or("")
    }

    /// All Devanagari words whose (exact) roman spelling is `roman`.
    pub fn lookup_exact(&self, roman: &str) -> Vec<String> {
        let key = roman.to_ascii_lowercase();
        let (lo, hi) = self.range(key.as_bytes());
        let mut out = Vec::new();
        for i in lo..hi {
            if self.roman_at(i) == key.as_bytes() {
                out.push(self.dev_at(i).to_string());
            }
        }
        out
    }

    /// Distinct Devanagari words whose roman spelling starts with `prefix`,
    /// ranked by the number of matching roman spellings, capped at `limit`.
    pub fn prefix_matches(&self, prefix: &str, limit: usize) -> Vec<(String, u32)> {
        let key = prefix.to_ascii_lowercase();
        if key.is_empty() || limit == 0 {
            return Vec::new();
        }
        let (lo, hi) = self.range(key.as_bytes());
        let mut counts: HashMap<String, u32> = HashMap::new();
        for i in lo..hi {
            if !self.roman_at(i).starts_with(key.as_bytes()) {
                break;
            }
            let d = self.dev_at(i).to_string();
            *counts.entry(d).or_insert(0) += 1;
            if counts.len() > limit * 4 {
                break;
            }
        }
        let mut v: Vec<(String, u32)> = counts.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v.truncate(limit);
        v
    }

    /// Size of the lexicon.
    pub fn len(&self) -> usize {
        self.roman_off.len()
    }

    /// Binary-search the range of entries whose roman starts with `key`.
    fn range(&self, key: &[u8]) -> (usize, usize) {
        let lo = self.partition_point(key);
        let mut hi = lo;
        while hi < self.len() && self.roman_at(hi).starts_with(key) {
            hi += 1;
        }
        (lo, hi)
    }

    /// First index where roman >= key (binary search over the arena).
    fn partition_point(&self, key: &[u8]) -> usize {
        let mut lo = 0usize;
        let mut hi = self.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.roman_at(mid) < key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Serialise to `path` with bincode.
    pub fn save(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::BufWriter;
        let file = std::fs::File::create(path)?;
        let mut w = BufWriter::new(file);
        bincode::serialize_into(&mut w, self)?;
        Ok(())
    }

    /// Load a bincode-serialised lexicon.
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        use std::io::BufReader;
        let file = std::fs::File::open(path)?;
        let r = BufReader::new(file);
        let l: Self = bincode::deserialize_from(r)?;
        Ok(l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_and_lookups_exact() {
        let lx = RomanLexicon::build(vec![
            ("namaste".to_string(), "नमस्ते".to_string()),
            ("nepal".to_string(), "नेपाल".to_string()),
            ("ka".to_string(), "क".to_string()),
        ]);
        assert_eq!(lx.lookup_exact("namaste"), vec!["नमस्ते"]);
        assert!(lx.lookup_exact("xyz").is_empty());
    }

    #[test]
    fn prefix_matching_is_case_insensitive_and_aggregates() {
        let lx = RomanLexicon::build(vec![
            ("nepal".to_string(), "नेपाल".to_string()),
            ("nepaal".to_string(), "नेपाल".to_string()),
            ("nepali".to_string(), "नेपाली".to_string()),
            ("Nepal".to_string(), "नेपाल".to_string()),
        ]);
        let m = lx.prefix_matches("nep", 10);
        // Two distinct roman spellings ("nepal" and "nepaal") both start with "nep".
        assert!(m.iter().any(|(d, c)| d == "नेपाल" && *c == 2));
        assert!(m.iter().any(|(d, _)| d == "नेपाली"));
    }

    #[test]
    fn save_load_roundtrip() {
        let lx = RomanLexicon::build(vec![("ka".to_string(), "क".to_string())]);
        let path = std::env::temp_dir().join("akshar_lex_test.bin");
        lx.save(&path).unwrap();
        let loaded = RomanLexicon::load(&path).unwrap();
        assert_eq!(loaded.lookup_exact("ka"), vec!["क"]);
        let _ = std::fs::remove_file(&path);
    }
}
