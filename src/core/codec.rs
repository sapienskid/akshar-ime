// File: src/core/codec.rs
//
// Compact wire encoding for the model container.
//
// The in-memory model is tuned for decode speed: `Vec<Vec<(u32, f32)>>` with
// direct indexing.  That layout is a poor thing to *store*, because it spends
// 8 bytes on every transition — a 32-bit id where the delta to the previous
// id needs ~11 bits, and a 32-bit float where the weight needs 8.  The
// trigram LM alone pays 15.64 MB for 2,049,828 transitions.
//
// This module is the translation layer between the two.  Nothing here is on
// the decode path; it runs once at save and once at load.  Keeping the wire
// format separate from the runtime layout means the decoder keeps its fast
// direct-indexed tables while the file gets small.
//
// Three techniques, all measured before adoption (see quantize_model):
//
//   * CSR — one flat id/weight array plus row offsets, instead of a `Vec` per
//     row with its own length prefix and heap pointer.
//   * Delta varints — successor ids sorted ascending and stored as
//     first differences, which are small and therefore short.
//   * Codebook quantization — each weight becomes a 1-byte index into a
//     256-entry Lloyd-Max codebook fitted to that table's own distribution.
//     Measured cost: mean |error| 0.011-0.031 nats on weights the decoder
//     clips at 25-30, worth about -0.1pp end to end.
//
// Byte order is little-endian throughout, matching bincode's default so the
// two halves of the container agree.

// ---------------------------------------------------------------------------
// LEB128 varints
// ---------------------------------------------------------------------------

/// Append `v` as an unsigned LEB128 varint.
#[inline]
pub fn write_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// Read a varint from `buf` at `*pos`, advancing it.
#[inline]
pub fn read_varint(buf: &[u8], pos: &mut usize) -> Result<u64, CodecError> {
    let mut v = 0u64;
    let mut shift = 0u32;
    loop {
        let b = *buf.get(*pos).ok_or(CodecError::Truncated)?;
        *pos += 1;
        // 10 groups of 7 bits is the most a u64 can occupy; beyond that the
        // stream is corrupt and shifting would panic in debug builds.
        if shift >= 64 {
            return Err(CodecError::Malformed("varint too long"));
        }
        v |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok(v);
        }
        shift += 7;
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum CodecError {
    Truncated,
    Malformed(&'static str),
}

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodecError::Truncated => write!(f, "unexpected end of encoded section"),
            CodecError::Malformed(m) => write!(f, "malformed encoded section: {m}"),
        }
    }
}

impl std::error::Error for CodecError {}

// ---------------------------------------------------------------------------
// Codebook quantization
// ---------------------------------------------------------------------------

/// Maximum codebook entries.  256 keeps indices in a single byte.
pub const CODEBOOK_MAX: usize = 256;

/// Fit a Lloyd-Max codebook to `values`.
///
/// Initialised on quantiles rather than a linear span of [min, max]: these are
/// `-log` probabilities, heavily skewed, and a uniform grid spends most of its
/// levels on a sparse tail while crushing the dense region where ranking
/// decisions are actually made.
pub fn fit_codebook(values: &[f32], levels: usize) -> Vec<f32> {
    if values.is_empty() {
        return vec![0.0];
    }
    let levels = levels.clamp(1, CODEBOOK_MAX);
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));

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

    for _ in 0..12 {
        let mut sums = vec![0.0f64; centroids.len()];
        let mut counts = vec![0usize; centroids.len()];
        for &v in &sorted {
            let i = nearest(&centroids, v);
            sums[i] += v as f64;
            counts[i] += 1;
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

/// Index of the nearest centroid.  Centroids are sorted, so this is a binary
/// search plus a comparison against the two straddling entries.
#[inline]
pub fn nearest(centroids: &[f32], v: f32) -> usize {
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

fn write_codebook(out: &mut Vec<u8>, codebook: &[f32]) {
    write_varint(out, codebook.len() as u64);
    for &c in codebook {
        out.extend_from_slice(&c.to_le_bytes());
    }
}

fn read_codebook(buf: &[u8], pos: &mut usize) -> Result<Vec<f32>, CodecError> {
    let n = read_varint(buf, pos)? as usize;
    if n > CODEBOOK_MAX {
        return Err(CodecError::Malformed("codebook larger than 256 entries"));
    }
    let mut cb = Vec::with_capacity(n);
    for _ in 0..n {
        let end = *pos + 4;
        let bytes = buf.get(*pos..end).ok_or(CodecError::Truncated)?;
        cb.push(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
        *pos = end;
    }
    Ok(cb)
}

// ---------------------------------------------------------------------------
// Adjacency lists: Vec<Vec<(u32, f32)>>
// ---------------------------------------------------------------------------

/// Encode an adjacency list as CSR + delta varints + codebook indices.
///
/// Row order is preserved exactly; within a row the entries are sorted by id
/// so the deltas are positive and small.  Sorting by id also means the decoder
/// *could* binary-search a row, which the frequency-sorted layout could not.
pub fn encode_adjacency(rows: &[Vec<(u32, f32)>]) -> Vec<u8> {
    let weights: Vec<f32> = rows.iter().flatten().map(|(_, w)| *w).collect();
    let codebook = fit_codebook(&weights, CODEBOOK_MAX);

    let mut out = Vec::with_capacity(weights.len() * 3 + 1024);
    write_codebook(&mut out, &codebook);
    write_varint(&mut out, rows.len() as u64);

    let mut sorted_row: Vec<(u32, f32)> = Vec::new();
    for row in rows {
        write_varint(&mut out, row.len() as u64);
        sorted_row.clear();
        sorted_row.extend_from_slice(row);
        sorted_row.sort_unstable_by_key(|(id, _)| *id);

        let mut prev = 0u32;
        for &(id, w) in &sorted_row {
            // First entry stores the absolute id; the rest store the gap.
            write_varint(&mut out, (id - prev) as u64);
            prev = id;
            out.push(nearest(&codebook, w) as u8);
        }
    }
    out
}

/// Inverse of [`encode_adjacency`].
pub fn decode_adjacency(buf: &[u8]) -> Result<Vec<Vec<(u32, f32)>>, CodecError> {
    let mut pos = 0usize;
    let codebook = read_codebook(buf, &mut pos)?;
    if codebook.is_empty() {
        return Err(CodecError::Malformed("empty codebook"));
    }
    let n_rows = read_varint(buf, &mut pos)? as usize;
    let mut rows = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let len = read_varint(buf, &mut pos)? as usize;
        let mut row = Vec::with_capacity(len);
        let mut prev = 0u32;
        for _ in 0..len {
            let delta = read_varint(buf, &mut pos)? as u32;
            let id = prev
                .checked_add(delta)
                .ok_or(CodecError::Malformed("id overflow"))?;
            prev = id;
            let idx = *buf.get(pos).ok_or(CodecError::Truncated)? as usize;
            pos += 1;
            let w = *codebook
                .get(idx)
                .ok_or(CodecError::Malformed("codebook index out of range"))?;
            row.push((id, w));
        }
        rows.push(row);
    }
    Ok(rows)
}

// ---------------------------------------------------------------------------
// Dense weight arrays: Vec<f32>
// ---------------------------------------------------------------------------

/// Encode a dense weight vector as codebook indices, one byte each.
pub fn encode_weights(values: &[f32]) -> Vec<u8> {
    let codebook = fit_codebook(values, CODEBOOK_MAX);
    let mut out = Vec::with_capacity(values.len() + 1024);
    write_codebook(&mut out, &codebook);
    write_varint(&mut out, values.len() as u64);
    for &v in values {
        out.push(nearest(&codebook, v) as u8);
    }
    out
}

/// Inverse of [`encode_weights`].
pub fn decode_weights(buf: &[u8]) -> Result<Vec<f32>, CodecError> {
    let mut pos = 0usize;
    let codebook = read_codebook(buf, &mut pos)?;
    if codebook.is_empty() {
        return Err(CodecError::Malformed("empty codebook"));
    }
    let n = read_varint(buf, &mut pos)? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let idx = *buf.get(pos).ok_or(CodecError::Truncated)? as usize;
        pos += 1;
        out.push(
            *codebook
                .get(idx)
                .ok_or(CodecError::Malformed("codebook index out of range"))?,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Id pair lists: Vec<(u32, u32)> (trigram context keys)
// ---------------------------------------------------------------------------

/// Encode context keys.
///
/// The keys index parallel arrays (`trigrams`, `trigram_backoff`), so their
/// order is load-bearing and cannot be changed.  But they can still be *stored*
/// in sorted order alongside a permutation that restores the original — and
/// sorted keys delta-encode to roughly a byte each instead of the two to three
/// an absolute id costs.
///
/// That trade only pays when the permutation is cheaper than what sorting
/// saves, which depends on how close to sorted the keys already are.  Both
/// layouts are produced and the smaller one is written, tagged so the decoder
/// knows which it got.  For an already-sorted table the permutation is skipped
/// entirely.
pub fn encode_pairs(pairs: &[(u32, u32)]) -> Vec<u8> {
    let plain = encode_pairs_plain(pairs);

    let is_sorted = pairs.windows(2).all(|w| w[0] <= w[1]);
    let delta = if is_sorted {
        // Already in order: delta-encode directly, no permutation needed.
        let mut out = vec![TAG_PAIRS_SORTED];
        write_pairs_delta(&mut out, pairs).map(|()| out)
    } else {
        let mut order: Vec<u32> = (0..pairs.len() as u32).collect();
        order.sort_unstable_by_key(|&i| pairs[i as usize]);
        let sorted: Vec<(u32, u32)> = order.iter().map(|&i| pairs[i as usize]).collect();

        let mut out = vec![TAG_PAIRS_PERMUTED];
        write_pairs_delta(&mut out, &sorted).map(|()| {
            for &i in &order {
                write_varint(&mut out, i as u64);
            }
            out
        })
    };

    match delta {
        Some(d) if d.len() < plain.len() => d,
        _ => plain,
    }
}

const TAG_PAIRS_PLAIN: u8 = 0;
const TAG_PAIRS_SORTED: u8 = 1;
const TAG_PAIRS_PERMUTED: u8 = 2;

fn encode_pairs_plain(pairs: &[(u32, u32)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pairs.len() * 3 + 16);
    out.push(TAG_PAIRS_PLAIN);
    write_varint(&mut out, pairs.len() as u64);
    for &(a, b) in pairs {
        write_varint(&mut out, a as u64);
        write_varint(&mut out, b as u64);
    }
    out
}

/// Delta-encode lexicographically ascending pairs: the gap in `a`, then `b`
/// itself, or its gap within a run of equal `a`.
///
/// Returns None if the input is not actually ascending. Both deltas would
/// underflow on unsigned subtraction and silently encode as maximum-width
/// varints — larger output and, for `b`, a corrupt round trip. Callers fall
/// back to the plain layout.
fn write_pairs_delta(out: &mut Vec<u8>, sorted: &[(u32, u32)]) -> Option<()> {
    write_varint(out, sorted.len() as u64);
    let (mut pa, mut pb) = (0u32, 0u32);
    for &(a, b) in sorted {
        write_varint(out, a.checked_sub(pa)? as u64);
        if a == pa {
            write_varint(out, b.checked_sub(pb)? as u64);
        } else {
            write_varint(out, b as u64);
        }
        pa = a;
        pb = b;
    }
    Some(())
}

/// Inverse of [`encode_pairs`].
pub fn decode_pairs(buf: &[u8]) -> Result<Vec<(u32, u32)>, CodecError> {
    let mut pos = 0usize;
    let tag = *buf.first().ok_or(CodecError::Truncated)?;
    pos += 1;

    match tag {
        TAG_PAIRS_PLAIN => {
            let n = read_varint(buf, &mut pos)? as usize;
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                let a = read_varint(buf, &mut pos)? as u32;
                let b = read_varint(buf, &mut pos)? as u32;
                out.push((a, b));
            }
            Ok(out)
        }
        TAG_PAIRS_SORTED => read_pairs_delta(buf, &mut pos),
        TAG_PAIRS_PERMUTED => {
            let sorted = read_pairs_delta(buf, &mut pos)?;
            let mut out = vec![(0u32, 0u32); sorted.len()];
            for &pair in &sorted {
                let i = read_varint(buf, &mut pos)? as usize;
                *out.get_mut(i)
                    .ok_or(CodecError::Malformed("permutation index out of range"))? = pair;
            }
            Ok(out)
        }
        _ => Err(CodecError::Malformed("unknown pair encoding tag")),
    }
}

fn read_pairs_delta(buf: &[u8], pos: &mut usize) -> Result<Vec<(u32, u32)>, CodecError> {
    let n = read_varint(buf, pos)? as usize;
    let mut out = Vec::with_capacity(n);
    let (mut pa, mut pb) = (0u32, 0u32);
    for _ in 0..n {
        let da = read_varint(buf, pos)? as u32;
        let a = pa
            .checked_add(da)
            .ok_or(CodecError::Malformed("pair id overflow"))?;
        let raw_b = read_varint(buf, pos)? as u32;
        let b = if a == pa {
            pb.checked_add(raw_b)
                .ok_or(CodecError::Malformed("pair id overflow"))?
        } else {
            raw_b
        };
        out.push((a, b));
        pa = a;
        pb = b;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Vocabulary: HashMap<String, u32>
// ---------------------------------------------------------------------------

// bincode stores this as 8 bytes of length + UTF-8 + 4 bytes of frequency:
// 37.1 bytes per word across the 470,012-word vocabulary, 16.64 MB in total,
// and 11.25 MB of that is raw UTF-8 text at 3 bytes per Devanagari codepoint.
//
// Two observations shrink it.  A word is a sequence of aksharas, and the model
// knows only ~5,941 of them after pruning, so an akshara is a short varint
// rather than 3-9 bytes of UTF-8.  And sorted Devanagari vocabularies share
// long prefixes, so each entry need only store what it does not share with its
// predecessor (front coding).
//
// Words containing an akshara the model does not know are kept verbatim in a
// literal section.  So are words whose segmentation does not rejoin to the
// original string: `segment` is believed to be an exact partition, but the
// vocabulary must round trip byte for byte regardless, so the encoder checks
// rather than assumes.

/// Encode a vocabulary against an akshara table.
pub fn encode_vocab(
    vocab: &std::collections::HashMap<String, u32>,
    aksharas: &[String],
) -> Vec<u8> {
    let index: std::collections::HashMap<&str, u32> = aksharas
        .iter()
        .enumerate()
        .map(|(i, a)| (a.as_str(), i as u32))
        .collect();

    let mut coded: Vec<(Vec<u32>, u32)> = Vec::with_capacity(vocab.len());
    let mut literal: Vec<(&str, u32)> = Vec::new();

    for (word, &freq) in vocab {
        match to_ids(word, &index) {
            Some(ids) => coded.push((ids, freq)),
            None => literal.push((word.as_str(), freq)),
        }
    }
    // Sort so consecutive entries share prefixes; also makes the output
    // deterministic, which HashMap iteration order is not.
    coded.sort_unstable();
    literal.sort_unstable();

    let mut out = Vec::with_capacity(coded.len() * 4 + literal.len() * 16 + 32);

    write_varint(&mut out, coded.len() as u64);
    let mut prev: &[u32] = &[];
    for (ids, freq) in &coded {
        let shared = prev.iter().zip(ids).take_while(|(a, b)| a == b).count();
        write_varint(&mut out, shared as u64);
        write_varint(&mut out, (ids.len() - shared) as u64);
        for &id in &ids[shared..] {
            write_varint(&mut out, id as u64);
        }
        write_varint(&mut out, *freq as u64);
        prev = ids;
    }

    write_varint(&mut out, literal.len() as u64);
    for (word, freq) in &literal {
        write_varint(&mut out, word.len() as u64);
        out.extend_from_slice(word.as_bytes());
        write_varint(&mut out, *freq as u64);
    }
    out
}

/// Segment `word` into akshara ids, or None if that would not round trip.
fn to_ids(word: &str, index: &std::collections::HashMap<&str, u32>) -> Option<Vec<u32>> {
    let units = crate::core::akshara::segment(word);
    if units.is_empty() {
        return None;
    }
    // Guard: the encoding reconstructs the word by concatenating akshara
    // strings, so anything that does not rejoin exactly must go to literals.
    if units.concat() != word {
        return None;
    }
    units
        .iter()
        .map(|u| index.get(u.as_str()).copied())
        .collect()
}

/// Inverse of [`encode_vocab`].
pub fn decode_vocab(
    buf: &[u8],
    aksharas: &[String],
) -> Result<std::collections::HashMap<String, u32>, CodecError> {
    let mut pos = 0usize;
    let n_coded = read_varint(buf, &mut pos)? as usize;
    let mut out = std::collections::HashMap::with_capacity(n_coded);

    let mut prev: Vec<u32> = Vec::new();
    for _ in 0..n_coded {
        let shared = read_varint(buf, &mut pos)? as usize;
        let rest = read_varint(buf, &mut pos)? as usize;
        if shared > prev.len() {
            return Err(CodecError::Malformed(
                "front-coding prefix exceeds previous word",
            ));
        }
        let mut ids = Vec::with_capacity(shared + rest);
        ids.extend_from_slice(&prev[..shared]);
        for _ in 0..rest {
            ids.push(read_varint(buf, &mut pos)? as u32);
        }
        let freq = read_varint(buf, &mut pos)? as u32;

        let mut word = String::new();
        for &id in &ids {
            word.push_str(
                aksharas
                    .get(id as usize)
                    .ok_or(CodecError::Malformed("akshara id out of range"))?,
            );
        }
        out.insert(word, freq);
        prev = ids;
    }

    let n_literal = read_varint(buf, &mut pos)? as usize;
    for _ in 0..n_literal {
        let len = read_varint(buf, &mut pos)? as usize;
        let end = pos + len;
        let bytes = buf.get(pos..end).ok_or(CodecError::Truncated)?;
        pos = end;
        let word = std::str::from_utf8(bytes)
            .map_err(|_| CodecError::Malformed("literal word is not UTF-8"))?
            .to_string();
        let freq = read_varint(buf, &mut pos)? as u32;
        out.insert(word, freq);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Roman chunk vocabulary: Vec<String>
// ---------------------------------------------------------------------------

// 100,578 chunks cost 1.18 MB as a `Vec<String>` — an 8-byte length prefix on
// strings averaging ~3.3 bytes.  A chunk is documented as 1..=5 lowercase
// ASCII letters, which `translit_model::pack_chunk_bytes` already packs into a
// u32 (5 bits per letter plus a length field).
//
// Chunk ids index this list positionally, so the order is load-bearing and the
// list cannot be sorted for front coding.  Packing is the available win.
// `pack_chunk_bytes` maps anything outside a-z to an escape code that does not
// survive unpacking, so each chunk is round-tripped at encode time and any
// that does not reproduce itself is stored as a literal instead.

/// Encode the chunk vocabulary, packing what fits into a u32.
pub fn encode_chunks(chunks: &[String]) -> Vec<u8> {
    use crate::core::translit_model::{pack_chunk_bytes, unpack_chunk};

    let mut out = Vec::with_capacity(chunks.len() * 5 + 16);
    write_varint(&mut out, chunks.len() as u64);
    for c in chunks {
        let packed = pack_chunk_bytes(c.as_bytes());
        if unpack_chunk(packed) == *c {
            out.push(0u8);
            out.extend_from_slice(&packed.to_le_bytes());
        } else {
            // Escape: anything the packing cannot represent exactly.
            out.push(1u8);
            write_varint(&mut out, c.len() as u64);
            out.extend_from_slice(c.as_bytes());
        }
    }
    out
}

/// Inverse of [`encode_chunks`].
pub fn decode_chunks(buf: &[u8]) -> Result<Vec<String>, CodecError> {
    use crate::core::translit_model::unpack_chunk;

    let mut pos = 0usize;
    let n = read_varint(buf, &mut pos)? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let tag = *buf.get(pos).ok_or(CodecError::Truncated)?;
        pos += 1;
        match tag {
            0 => {
                let end = pos + 4;
                let b = buf.get(pos..end).ok_or(CodecError::Truncated)?;
                pos = end;
                out.push(unpack_chunk(u32::from_le_bytes([b[0], b[1], b[2], b[3]])));
            }
            1 => {
                let len = read_varint(buf, &mut pos)? as usize;
                let end = pos + len;
                let b = buf.get(pos..end).ok_or(CodecError::Truncated)?;
                pos = end;
                out.push(
                    std::str::from_utf8(b)
                        .map_err(|_| CodecError::Malformed("literal chunk is not UTF-8"))?
                        .to_string(),
                );
            }
            _ => return Err(CodecError::Malformed("unknown chunk tag")),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_round_trips_across_widths() {
        for v in [
            0u64,
            1,
            127,
            128,
            300,
            16_383,
            16_384,
            u32::MAX as u64,
            u64::MAX,
        ] {
            let mut buf = Vec::new();
            write_varint(&mut buf, v);
            let mut pos = 0;
            assert_eq!(read_varint(&buf, &mut pos).unwrap(), v, "value {v}");
            assert_eq!(pos, buf.len(), "value {v} left trailing bytes");
        }
    }

    #[test]
    fn varint_rejects_truncated_and_overlong() {
        let mut pos = 0;
        assert_eq!(read_varint(&[0x80], &mut pos), Err(CodecError::Truncated));
        let mut pos = 0;
        assert_eq!(
            read_varint(&[0x80; 12], &mut pos),
            Err(CodecError::Malformed("varint too long"))
        );
    }

    #[test]
    fn adjacency_round_trips_with_ids_preserved_exactly() {
        let rows = vec![
            vec![(5u32, 1.25f32), (1, 0.5), (900_000, 3.75)],
            vec![],
            vec![(0, 0.0)],
        ];
        let enc = encode_adjacency(&rows);
        let dec = decode_adjacency(&enc).unwrap();

        assert_eq!(dec.len(), rows.len());
        for (orig, back) in rows.iter().zip(&dec) {
            let mut want: Vec<u32> = orig.iter().map(|(i, _)| *i).collect();
            want.sort_unstable();
            let got: Vec<u32> = back.iter().map(|(i, _)| *i).collect();
            // Ids must be exact and ascending; only weights are lossy.
            assert_eq!(got, want);
        }
    }

    #[test]
    fn adjacency_weights_survive_within_quantization_error() {
        // More distinct weights than codebook entries, to exercise the lossy path.
        let row: Vec<(u32, f32)> = (0..1000u32).map(|i| (i, i as f32 * 0.01)).collect();
        let dec = decode_adjacency(&encode_adjacency(std::slice::from_ref(&row))).unwrap();
        for (&(_, want), &(_, got)) in row.iter().zip(&dec[0]) {
            assert!((want - got).abs() < 0.2, "want {want} got {got}");
        }
    }

    #[test]
    fn small_value_sets_are_lossless() {
        // Fewer distinct values than codebook slots: quantization must be exact.
        let rows = vec![(0..500u32)
            .map(|i| (i, [0.5f32, 1.5, 2.5][i as usize % 3]))
            .collect()];
        let dec = decode_adjacency(&encode_adjacency(&rows)).unwrap();
        for (&(_, want), &(_, got)) in rows[0].iter().zip(&dec[0]) {
            assert_eq!(want, got);
        }
    }

    #[test]
    fn weights_round_trip() {
        let vals: Vec<f32> = (0..300).map(|i| (i as f32).sin() * 5.0).collect();
        let dec = decode_weights(&encode_weights(&vals)).unwrap();
        assert_eq!(dec.len(), vals.len());
        for (w, g) in vals.iter().zip(&dec) {
            assert!((w - g).abs() < 0.1);
        }
    }

    #[test]
    fn pairs_round_trip_preserving_order() {
        let pairs = vec![(3u32, 9u32), (0, 0), (1_000_000, 7)];
        assert_eq!(decode_pairs(&encode_pairs(&pairs)).unwrap(), pairs);
    }

    /// Order is load-bearing: the keys index parallel arrays, so an encoding
    /// that sorted them without restoring the permutation would silently
    /// misalign every trigram context with its successors and backoff.
    #[test]
    fn pairs_preserve_order_when_unsorted_and_sorted() {
        let unsorted = vec![(5u32, 1u32), (0, 9), (5, 0), (2, 2)];
        assert_eq!(decode_pairs(&encode_pairs(&unsorted)).unwrap(), unsorted);

        let sorted: Vec<(u32, u32)> = (0..500u32).map(|i| (i / 3, i % 7)).collect();
        assert_eq!(decode_pairs(&encode_pairs(&sorted)).unwrap(), sorted);
    }

    /// A large lexicographically ascending table should pick the delta layout
    /// and beat the plain one; that is the point of the alternative encoding.
    #[test]
    fn sorted_pairs_encode_smaller_than_plain() {
        let sorted: Vec<(u32, u32)> = (0..5000u32).map(|i| (i / 4, i % 4)).collect();
        assert!(
            sorted.windows(2).all(|w| w[0] <= w[1]),
            "test data must be sorted"
        );
        let enc = encode_pairs(&sorted);
        assert_eq!(enc[0], TAG_PAIRS_SORTED, "expected the delta layout to win");
        assert!(
            enc.len() < encode_pairs_plain(&sorted).len(),
            "delta {} vs plain {}",
            enc.len(),
            encode_pairs_plain(&sorted).len()
        );
        assert_eq!(decode_pairs(&enc).unwrap(), sorted);
    }

    #[test]
    fn decoders_reject_garbage_without_panicking() {
        assert!(decode_adjacency(&[0xFF, 0xFF]).is_err());
        assert!(decode_weights(&[]).is_err());
        assert!(decode_pairs(&[0xFF]).is_err());
    }

    #[test]
    fn vocab_round_trips_exactly() {
        let aksharas: Vec<String> = ["क", "म", "ल", "ा", "ने", "पा"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut vocab = std::collections::HashMap::new();
        for (w, f) in [("कमल", 12u32), ("कम", 5), ("नेपाल", 900), ("क", 1)] {
            vocab.insert(w.to_string(), f);
        }
        let back = decode_vocab(&encode_vocab(&vocab, &aksharas), &aksharas).unwrap();
        assert_eq!(back, vocab, "vocabulary must round trip byte for byte");
    }

    /// Words the akshara table cannot express must survive via the literal
    /// section rather than being dropped or corrupted.
    #[test]
    fn vocab_keeps_unrepresentable_words() {
        let aksharas = vec!["क".to_string()];
        let mut vocab = std::collections::HashMap::new();
        vocab.insert("क".to_string(), 3u32);
        vocab.insert("नेपाल".to_string(), 7u32); // no ids for these aksharas
        vocab.insert("hello".to_string(), 1u32); // not Devanagari at all
        let back = decode_vocab(&encode_vocab(&vocab, &aksharas), &aksharas).unwrap();
        assert_eq!(back, vocab);
    }

    #[test]
    fn vocab_decoder_rejects_bad_prefix() {
        // shared=5 on the first entry, where there is no previous word.
        let bad = {
            let mut v = Vec::new();
            write_varint(&mut v, 1);
            write_varint(&mut v, 5);
            v
        };
        assert!(decode_vocab(&bad, &[]).is_err());
    }

    #[test]
    fn chunks_round_trip_including_unpackable() {
        let chunks: Vec<String> = ["ka", "krya", "a", "", "sTe", "toolongchunk", "ñ"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let back = decode_chunks(&encode_chunks(&chunks)).unwrap();
        assert_eq!(
            back, chunks,
            "chunk ids are positional; order and content must be exact"
        );
    }

    #[test]
    fn empty_inputs_round_trip() {
        assert!(decode_adjacency(&encode_adjacency(&[])).unwrap().is_empty());
        assert!(decode_weights(&encode_weights(&[])).unwrap().is_empty());
        assert!(decode_pairs(&encode_pairs(&[])).unwrap().is_empty());
    }
}
