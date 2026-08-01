// File: src/core/normalizer.rs
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// A normalized Roman input variant with an associated penalty.
/// Lower penalty means a more trusted rewrite.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RomanVariant {
    pub roman: String,
    pub penalty: u64,
}

#[derive(Debug, Clone, Copy)]
struct RewriteRule {
    from: &'static str,
    to: &'static str,
    cost: u64,
}

/// Costed rewrite rules approximating a compact finite-state normalization layer.
const NORMALIZATION_RULES: &[RewriteRule] = &[
    // Long-vowel alternations.
    RewriteRule {
        from: "ee",
        to: "ii",
        cost: 1,
    },
    RewriteRule {
        from: "oo",
        to: "uu",
        cost: 1,
    },
    RewriteRule {
        from: "ou",
        to: "au",
        cost: 1,
    },
    RewriteRule {
        from: "ae",
        to: "ai",
        cost: 1,
    },
    // v/w interchange.
    RewriteRule {
        from: "w",
        to: "v",
        cost: 1,
    },
    RewriteRule {
        from: "W",
        to: "V",
        cost: 1,
    },
    // Optional separators frequently used by users while typing transliteration.
    RewriteRule {
        from: "-",
        to: "",
        cost: 1,
    },
    RewriteRule {
        from: "_",
        to: "",
        cost: 1,
    },
    RewriteRule {
        from: "'",
        to: "",
        cost: 1,
    },
    RewriteRule {
        from: " ",
        to: "",
        cost: 1,
    },
    // Compact repeated-vowel collapses.
    RewriteRule {
        from: "aaa",
        to: "aa",
        cost: 1,
    },
    RewriteRule {
        from: "eee",
        to: "ee",
        cost: 1,
    },
    RewriteRule {
        from: "iii",
        to: "ii",
        cost: 1,
    },
    RewriteRule {
        from: "ooo",
        to: "oo",
        cost: 1,
    },
    RewriteRule {
        from: "uuu",
        to: "uu",
        cost: 1,
    },
    // Optional long-vowel expansion to surface grammar-friendly alternatives.
    RewriteRule {
        from: "a",
        to: "aa",
        cost: 2,
    },
    RewriteRule {
        from: "i",
        to: "ii",
        cost: 2,
    },
    RewriteRule {
        from: "u",
        to: "uu",
        cost: 2,
    },
    // Conservative long-vowel simplification for noisy typing.
    RewriteRule {
        from: "aa",
        to: "a",
        cost: 2,
    },
    RewriteRule {
        from: "ii",
        to: "i",
        cost: 2,
    },
    RewriteRule {
        from: "uu",
        to: "u",
        cost: 2,
    },
];

fn is_aggressive_short_vowel_expansion(rule: &RewriteRule) -> bool {
    matches!(
        (rule.from, rule.to),
        ("a", "aa") | ("i", "ii") | ("u", "uu")
    )
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct SearchState {
    input_idx: usize,
    output: String,
    cost: u64,
}

impl Ord for SearchState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior by `cost`.
        other
            .cost
            .cmp(&self.cost)
            .then_with(|| other.output.len().cmp(&self.output.len()))
            .then_with(|| other.input_idx.cmp(&self.input_idx))
    }
}

impl PartialOrd for SearchState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Expand the user query into weighted normalized variants via transducer search.
pub fn expand_query_variants(input: &str, max_variants: usize) -> Vec<RomanVariant> {
    let max_variants = max_variants.max(1);
    let max_expansions = max_variants.saturating_mul(128);
    let short_input = input.chars().count() <= 2;

    let mut heap = BinaryHeap::new();
    heap.push(SearchState {
        input_idx: 0,
        output: String::with_capacity(input.len()),
        cost: 0,
    });

    // Best known cost for a specific (input_idx, output) state.
    let mut best_cost_by_state: HashMap<(usize, String), u64> = HashMap::new();
    let mut best_cost_by_output: HashMap<String, u64> = HashMap::new();
    let mut expansions = 0usize;

    while let Some(state) = heap.pop() {
        if expansions >= max_expansions {
            break;
        }
        expansions += 1;

        let state_key = (state.input_idx, state.output.clone());
        if let Some(&best) = best_cost_by_state.get(&state_key) {
            if state.cost > best {
                continue;
            }
        }
        best_cost_by_state.insert(state_key, state.cost);

        if state.input_idx == input.len() {
            best_cost_by_output
                .entry(state.output.clone())
                .and_modify(|best| *best = (*best).min(state.cost))
                .or_insert(state.cost);
            continue;
        }

        // Default transducer transition: copy one input character.
        if let Some(ch) = input[state.input_idx..].chars().next() {
            let mut copied = state.clone();
            copied.input_idx += ch.len_utf8();
            copied.output.push(ch);
            heap.push(copied);
        }

        // Costed rewrite transitions.
        for rule in NORMALIZATION_RULES {
            if short_input && is_aggressive_short_vowel_expansion(rule) {
                continue;
            }
            if input[state.input_idx..].starts_with(rule.from) {
                let mut rewritten = state.clone();
                rewritten.input_idx += rule.from.len();
                rewritten.output.push_str(rule.to);
                rewritten.cost = rewritten.cost.saturating_add(rule.cost);
                heap.push(rewritten);
            }
        }
    }

    if !best_cost_by_output.contains_key(input) {
        best_cost_by_output.insert(input.to_string(), 0);
    }

    let mut variants: Vec<RomanVariant> = best_cost_by_output
        .into_iter()
        .map(|(roman, penalty)| RomanVariant { roman, penalty })
        .collect();

    variants.sort_by(|a, b| {
        a.penalty
            .cmp(&b.penalty)
            .then_with(|| b.roman.len().cmp(&a.roman.len()))
            .then_with(|| a.roman.cmp(&b.roman))
    });
    variants.truncate(max_variants);
    variants
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizer_keeps_input_as_best_cost() {
        let variants = expand_query_variants("namaste", 5);
        assert!(!variants.is_empty());
        assert_eq!(variants[0].roman, "namaste");
        assert_eq!(variants[0].penalty, 0);
    }

    #[test]
    fn normalizer_applies_weighted_separator_rule() {
        let variants = expand_query_variants("k-a_l", 8);
        assert!(variants.iter().any(|v| v.roman == "kal"));
    }

    #[test]
    fn normalizer_applies_rewrite_rules() {
        let variants = expand_query_variants("kyaee", 10);
        assert!(variants.iter().any(|v| v.roman == "kyaii"));
    }

    #[test]
    fn normalizer_supports_long_vowel_simplification() {
        let variants = expand_query_variants("malaai", 12);
        assert!(variants.iter().any(|v| v.roman == "malai"));
    }

    #[test]
    fn normalizer_supports_long_vowel_expansion() {
        let variants = expand_query_variants("maya", 16);
        assert!(variants.iter().any(|v| v.roman == "maaya"));
        assert!(variants.iter().any(|v| v.roman == "maayaa"));
    }

    #[test]
    fn normalizer_avoids_short_vowel_expansion_noise() {
        let variants = expand_query_variants("ic", 12);
        assert!(!variants.iter().any(|v| v.roman == "iic"));
    }
}
