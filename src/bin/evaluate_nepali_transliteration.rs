use akshar_ime::ImeEngine;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Debug, Clone)]
struct EvalCase {
    roman: String,
    targets: Vec<String>,
}

#[derive(Debug, Clone)]
struct EvalResult {
    case: EvalCase,
    top_suggestions: Vec<String>,
    top1_hit: bool,
    topk_hit: bool,
}

fn main() {
    let mut dataset_path = "data/eval/aksharantar_test.tsv".to_string();
    let mut topk = 5usize;
    let mut show_misses = 20usize;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dataset" => {
                if let Some(path) = args.next() {
                    dataset_path = path;
                } else {
                    eprintln!("Missing value for --dataset");
                    std::process::exit(2);
                }
            }
            "--topk" => {
                if let Some(value) = args.next() {
                    if let Ok(parsed) = value.parse::<usize>() {
                        topk = parsed.max(1);
                    } else {
                        eprintln!("Invalid --topk value: {value}");
                        std::process::exit(2);
                    }
                } else {
                    eprintln!("Missing value for --topk");
                    std::process::exit(2);
                }
            }
            "--show-misses" => {
                if let Some(value) = args.next() {
                    if let Ok(parsed) = value.parse::<usize>() {
                        show_misses = parsed;
                    } else {
                        eprintln!("Invalid --show-misses value: {value}");
                        std::process::exit(2);
                    }
                } else {
                    eprintln!("Missing value for --show-misses");
                    std::process::exit(2);
                }
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            other => {
                eprintln!("Unknown argument: {other}");
                print_help();
                std::process::exit(2);
            }
        }
    }

    let cases = match parse_dataset(&dataset_path) {
        Ok(cases) if !cases.is_empty() => cases,
        Ok(_) => {
            eprintln!("Dataset is empty: {dataset_path}");
            std::process::exit(2);
        }
        Err(err) => {
            eprintln!("Failed to parse dataset `{dataset_path}`: {err}");
            std::process::exit(1);
        }
    };

    let engine = ImeEngine::new();
    let mut top1_hits = 0usize;
    let mut topk_hits = 0usize;
    let mut results = Vec::with_capacity(cases.len());

    for case in cases {
        let suggestions = engine.get_suggestions(&case.roman, topk.max(8));
        let suggestion_words: Vec<String> = suggestions.into_iter().map(|(dev, _)| dev).collect();
        let top1_hit = suggestion_words
            .first()
            .is_some_and(|first| case.targets.iter().any(|t| t == first));
        let topk_hit = suggestion_words
            .iter()
            .take(topk)
            .any(|s| case.targets.iter().any(|t| t == s));

        if top1_hit {
            top1_hits += 1;
        }
        if topk_hit {
            topk_hits += 1;
        }

        results.push(EvalResult {
            case,
            top_suggestions: suggestion_words,
            top1_hit,
            topk_hit,
        });
    }

    let total = results.len() as f64;
    let top1_acc = (top1_hits as f64 / total) * 100.0;
    let topk_acc = (topk_hits as f64 / total) * 100.0;

    println!("Aksharantar Nepali IME Evaluation (real held-out test split)");
    println!("Dataset: {dataset_path}");
    println!("Total cases: {}", results.len());
    println!(
        "Top-1 accuracy: {top1_hits}/{} ({top1_acc:.2}%)",
        results.len()
    );
    println!(
        "Top-{topk} accuracy: {topk_hits}/{} ({topk_acc:.2}%)",
        results.len()
    );

    if show_misses > 0 {
        println!("\nMisses (showing up to {show_misses}):");
        let mut shown = 0usize;
        for result in results.iter().filter(|r| !r.topk_hit) {
            if shown >= show_misses {
                break;
            }
            let top = result
                .top_suggestions
                .iter()
                .take(topk)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            println!(
                "  roman=`{}` targets={:?} top={} top1_hit={}",
                result.case.roman, result.case.targets, top, result.top1_hit
            );
            shown += 1;
        }
        if shown == 0 {
            println!("  none");
        }
    }
}

fn parse_dataset(path: &str) -> Result<Vec<EvalCase>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    let mut cases = Vec::new();

    for (line_no, line) in reader.lines().enumerate() {
        let line_no = line_no + 1;
        let line = line.map_err(|e| e.to_string())?;
        if let Some(case) = parse_dataset_line(&line, line_no)? {
            cases.push(case);
        }
    }
    Ok(cases)
}

fn parse_dataset_line(line: &str, line_no: usize) -> Result<Option<EvalCase>, String> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return Ok(None);
    }

    let (roman, targets_raw) = trimmed
        .split_once('\t')
        .ok_or_else(|| format!("line {line_no}: expected `roman<TAB>target1|target2`"))?;
    let roman = roman.trim().to_string();
    if roman.is_empty() {
        return Err(format!("line {line_no}: empty roman key"));
    }

    let targets: Vec<String> = targets_raw
        .split('|')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if targets.is_empty() {
        return Err(format!("line {line_no}: no targets found"));
    }

    Ok(Some(EvalCase { roman, targets }))
}

fn print_help() {
    println!("Usage: cargo run --bin evaluate_nepali_transliteration -- [options]");
    println!("Options:");
    println!(
        "  --dataset <path>       Dataset TSV path (default: data/eval/aksharantar_test.tsv)"
    );
    println!("  --topk <n>             Top-k for hit metric (default: 5)");
    println!("  --show-misses <n>      Number of misses to print (default: 20)");
    println!("  -h, --help             Show help");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_line_parser_handles_multiple_targets() {
        let line = "station\tस्टेशन|स्टेसन";
        let parsed = parse_dataset_line(line, 1).expect("line should parse");
        let case = parsed.expect("line should produce case");
        assert_eq!(case.roman, "station");
        assert_eq!(case.targets.len(), 2);
    }

    #[test]
    fn dataset_line_parser_skips_comments() {
        let parsed = parse_dataset_line("# comment", 1).expect("comment should parse");
        assert!(parsed.is_none());
    }
}
