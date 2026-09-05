#!/usr/bin/env python3
# tools/scrape_news.py — PERSONAL USE: harvest word/pair statistics from
# Devanagari news portals.  Raw text stays on your machine; only aggregate
# counts are produced (word_freq.tsv, word_pairs.tsv) for the IME pipeline.
#
# Usage: python3 tools/scrape_news.py --sitemap URL --out /tmp/news.txt \
#          [--max-pages 2000] [--delay 1.0]
#
# Respects robots.txt, rate-limits requests, keeps only Devanagari-rich lines.

import argparse
import re
import time
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from collections import Counter
from html import unescape

DEV = re.compile(r"[\u0900-\u097f]")
TAG = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.DOTALL)
PAIRS = Counter()
WORDS = Counter()
seen_urls = set()


def fetch(url, delay, robots):
    if robots and not robots.can_fetch("*", url):
        print(f"  robots.txt disallows {url}")
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "akshar-vocab-research/1.0 (personal)"})
        with urllib.request.urlopen(req, timeout=15) as r:
            time.sleep(delay)
            return r.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  skip {url}: {e}")
        return None


def harvest(html):
    text = TAG.sub(" ", unescape(html))
    prev_word = None
    for para in text.split("<p>") or [text]:
        for line in para.splitlines():
            if sum(1 for c in line if DEV.match(c)) < 10 or len(line) < 20:
                continue
            words = [
                w for w in re.findall(r"[\u0900-\u097f]+", line) if 1 < len(w) <= 24
            ]
            for w in words:
                WORDS[w] += 1
            for w in words:
                if prev_word is not None:
                    PAIRS[(prev_word, w)] += 1
                prev_word = w


def walk_sitemap(url, depth, max_pages, delay):
    if depth > 2 or len(seen_urls) >= max_pages:
        return
    body = fetch(url, delay, ROBOTS)
    if not body:
        return
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return
    ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    if root.tag.endswith("sitemapindex"):
        for loc in root.findall(".//s:loc", ns) or root.findall(".//loc"):
            walk_sitemap(loc.text.strip(), depth + 1, max_pages, delay)
    else:
        for loc in root.findall(".//s:loc", ns) or root.findall(".//loc"):
            u = loc.text.strip()
            if u in seen_urls or len(seen_urls) >= max_pages:
                continue
            seen_urls.add(u)
            page = fetch(u, delay, ROBOTS)
            if page:
                harvest(page)
                if len(seen_urls) % 50 == 0:
                    print(f"  {len(seen_urls)} pages, {len(WORDS)} words...")


def main():
    global ROBOTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--sitemap", required=True)
    ap.add_argument("--out", default="/tmp/news.txt")
    ap.add_argument("--max-pages", type=int, default=2000)
    ap.add_argument("--delay", type=float, default=1.0)
    a = ap.parse_args()

    origin = "/".join(a.sitemap.split("/")[:3])
    ROBOTS = urllib.robotparser.RobotFileParser()
    ROBOTS.set_url(origin + "/robots.txt")
    try:
        ROBOTS.read()
    except Exception:
        print("no robots.txt reachable; proceeding carefully")

    walk_sitemap(a.sitemap, 0, a.max_pages, a.delay)

    with open(a.out, "w", encoding="utf-8") as f:
        for w, c in WORDS.most_common():
            f.write((w + " ") * min(c, 50) + "\n")  # cap per-word expansion
    with open(a.out + ".pairs.tsv", "w", encoding="utf-8") as f:
        for (a, b), c in PAIRS.most_common():
            if c >= 3:
                f.write(f"{a}\t{b}\t{c}\n")
    print(f"done: {len(seen_urls)} pages, {len(WORDS)} words, {len(PAIRS)} pairs")
    print(f"raw lines -> {a.out} (personal); pair counts -> {a.out}.pairs.tsv")


if __name__ == "__main__":
    main()
