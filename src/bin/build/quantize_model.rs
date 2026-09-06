// File: src/bin/build/quantize_model.rs
//
// Measure the accuracy cost of 8-bit weight quantization *before* committing
// to a new on-disk format.
//
// The translit model is 29.42 MB raw and only compresses 2.07x under Brotli,
// because its payload is f32 -log probabilities — high-entropy noise in the
// low mantissa bits that no general-purpose compressor can exploit.  Replacing
// each weight with an 8-bit index into a per-table codebook is what makes that
// section small (and far more compressible).  The open question is what it
// costs in accuracy.
//
// This tool answers that question without touching the serialization: it
// quantizes every weight to its codebook centroid and writes the model back in
// the existing format.  File size is therefore unchanged — the point is purely
// to run the benchmarks against quantized weights and see whether the format
// work is worth doing.
//
// Codebooks are built by Lloyd-Max on the actual weight distribution rather
// than uniformly over [min, max]: -log probabilities are heavily skewed, and a
// uniform grid would spend most of its levels on a sparse tail while crushing
// the dense region where ranking decisions are actually made.
//
// Usage:
//   cargo run --release --bin quantize_model -- \
//     --model data/akshar.model --out data/akshar_quantized.model [--levels 256]

use akshar_ime::core::unified::UnifiedModel;
use std::path::Path;

/// Lloyd-Max scalar quantizer over `values`, returning the codebook.
fn build_codebook(values: &[f32], levels: usize, iters: usize) -> Vec<f32> {
    if values.is_empty() {
        return vec![0.0];
    }
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

    // Initialise on quantiles so every level starts with roughly equal mass.
    // A min/max linear init wastes levels on the tail from the first step and
    // Lloyd-Max does not always recover from that.
    let mut centroids: Vec<f32> = (0..levels)
        .map(|i| {
            let q = (i as f64 + 0.5) / levels as f64;
            sorted[((q * sorted.len() as f64) as usize).min(sorted.len() - 1)]
        })
        .collect();
    centroids.dedup();
    if centroids.len() < 2 {
        return centroids;
    }

    for _ in 0..iters {
        let mut sums = vec![0.0f64; centroids.len()];
        let mut counts = vec![0usize; centroids.len()];
        for &v in &sorted {
            let idx = nearest(&centroids, v);
            sums[idx] += v as f64;
            counts[idx] += 1;
        }
        let mut moved = false;
        for i in 0..centroids.len() {
            if counts[i] > 0 {
                let c = (sums[i] / counts[i] as f64) as f32;
                if c != centroids[i] {
                    centroids[i] = c;
                    moved = true;
                }
            }
        }
        if !moved {
            break;
        }
    }
    centroids
}

/// Nearest centroid by binary search — centroids stay sorted throughout.
#[inline]
fn nearest(centroids: &[f32], v: f32) -> usize {
    match centroids.binary_search_by(|c| c.total_cmp(&v)) {
        Ok(i) => i,
        Err(0) => 0,
        Err(i) if i >= centroids.len() => centroids.len() - 1,
        Err(i) => {
            if (v - centroids[i - 1]).abs() <= (centroids[i] - v).abs() {
                i - 1
            } else {
                i
            }
        }
    }
}

/// Quantize a weight slice in place, reporting mean absolute error.
fn quantize(name: &str, values: Vec<&mut f32>, levels: usize) {
    if values.is_empty() {
        println!("  {name:<18} (empty)");
        return;
    }
    let raw: Vec<f32> = values.iter().map(|v| **v).collect();
    let codebook = build_codebook(&raw, levels, 12);
    let mut err = 0.0f64;
    let mut max_err = 0.0f32;
    let n = values.len();
    for v in values {
        let q = codebook[nearest(&codebook, *v)];
        let e = (q - *v).abs();
        err += e as f64;
        max_err = max_err.max(e);
        *v = q;
    }
    println!(
        "  {name:<18} {n:>9} weights -> {:>3} levels   mean|err|={:.5} nats  max={:.5}",
        codebook.len(),
        err / n as f64,
        max_err
    );
}

fn main() {
    let mut model_path = "data/akshar.model".to_string();
    let mut out_path = "data/akshar_quantized.model".to_string();
    let mut levels = 256usize;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model_path = args.next().expect("value for --model"),
            "--out" => out_path = args.next().expect("value for --out"),
            "--levels" => levels = args.next().expect("value for --levels").parse().unwrap(),
            "--help" | "-h" => {
                println!(
                    "quantize_model — measure the accuracy cost of N-level weight quantization"
                );
                println!("  --model <path>   input unified model");
                println!("  --out <path>     output model with quantized weights");
                println!("  --levels <n>     quantization levels (default 256 = 8 bit)");
                return;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    let mut m = UnifiedModel::load(Path::new(&model_path)).expect("load model");
    println!("Quantizing {model_path} to {levels} levels per table\n");

    let t = &mut m.translit;
    quantize(
        "emissions",
        t.emissions.iter_mut().flatten().map(|(_, w)| w).collect(),
        levels,
    );
    quantize(
        "bigram LM",
        t.bigrams.iter_mut().flatten().map(|(_, w)| w).collect(),
        levels,
    );
    quantize(
        "trigram LM",
        t.trigrams.iter_mut().flatten().map(|(_, w)| w).collect(),
        levels,
    );
    quantize("bigram backoff", t.backoff.iter_mut().collect(), levels);
    quantize(
        "trigram backoff",
        t.trigram_backoff.iter_mut().collect(),
        levels,
    );
    quantize("unigram KN", t.unigram_kn.iter_mut().collect(), levels);
    quantize("word start", t.word_start.iter_mut().collect(), levels);

    m.save(Path::new(&out_path)).expect("save model");
    let size = std::fs::metadata(&out_path).expect("metadata").len();
    println!(
        "\nWrote {out_path} ({:.2} MB in the CURRENT format — unchanged by design; \n\
         the saving comes from storing 1-byte codebook indices instead of f32).",
        size as f64 / (1024.0 * 1024.0)
    );
}
