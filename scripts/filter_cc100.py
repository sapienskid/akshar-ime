#!/usr/bin/env python3
# Filter a CC100 raw text stream down to Devanagari-heavy lines.
# Usage: filter_cc100.py <in.txt> <out.txt> [max_lines]
import lzma
import sys

in_path, out_path = sys.argv[1], sys.argv[2]
max_lines = int(sys.argv[3]) if len(sys.argv) > 3 else 3_000_000

opener = lzma.open if in_path.endswith('.xz') else open
n = 0
with opener(in_path, 'rt', encoding='utf-8', errors='ignore') as f, \
        open(out_path, 'w', encoding='utf-8') as out:
    for line in f:
        dev = sum(1 for c in line if '\u0900' <= c <= '\u097f')
        if dev >= 10 and len(line.strip()) > 20:
            out.write(line)
            n += 1
            if n >= max_lines:
                break
print(f"kept {n} lines -> {out_path}")
