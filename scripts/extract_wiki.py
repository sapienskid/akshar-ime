#!/usr/bin/env python3
# Extract clean Devanagari text lines from a Nepali Wikipedia XML dump (bz2).
# Usage: extract_wiki.py <dump.xml.bz2> <out.txt>
# (Same recipe as the release build — see .github/workflows/release.yml.)
import bz2
import re
import sys

dump_path, out_path = sys.argv[1], sys.argv[2]
text_re = re.compile(r'<text[^>]*>(.*?)</text>', re.DOTALL)
tag_re = re.compile(r'\[\[([^|\]]*\|)?([^|\]]*)\]\]')
cleanup = re.compile(r'\{\{.*?\}\}|<[^>]+>|\[http\S*\]')

n = 0
with bz2.open(dump_path, 'rt', encoding='utf-8', errors='ignore') as f, \
        open(out_path, 'w', encoding='utf-8') as out:
    data = f.read()
    for m in text_re.finditer(data):
        t = tag_re.sub(r'\2', m.group(1))
        t = cleanup.sub(' ', t)
        for line in t.splitlines():
            dev = sum(1 for c in line if '\u0900' <= c <= '\u097f')
            if dev >= 10:
                out.write(line + '\n')
                n += 1
print(f"extracted {n} lines -> {out_path}")
