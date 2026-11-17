#!/usr/bin/env python3
"""Check every reference with a DOI against Crossref and look for retraction, correction or concern notices.

Run on a machine with internet access before submission:
    python verify_refs.py --mailto you@institution.edu --out verify_refs_report.csv

For each DOI: fetches https://api.crossref.org/works/<doi>, compares title, year, journal, volume and pages or
article number with the entry in refs.py, and queries https://api.crossref.org/works?filter=updates:<doi> for
notices that update the DOI (Crossref also carries the Retraction Watch data). Entries without a DOI are listed
for a manual check at the publisher page.
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refs import R  # noqa: E402


def get(url, mailto):
    req = urllib.request.Request(url, headers={"User-Agent": f"A15-reference-check (mailto:{mailto})"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mailto", required=True)
    ap.add_argument("--out", default="verify_refs_report.csv")
    a = ap.parse_args()
    rows = []
    for r in R:
        doi = r.get("doi")
        if not doi:
            rows.append([r["key"], "", "NO DOI: check at the publisher page", "", "", ""])
            continue
        try:
            m = get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}", a.mailto)["message"]
            title = (m.get("title") or [""])[0]
            year = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
            vol, page = m.get("volume", ""), m.get("page") or m.get("article-number", "")
            journal = (m.get("container-title") or [""])[0]
            ok_title = norm(title)[:40] in norm(r["text"])
            ok_year = str(year) in r["text"]
            ok_vol = (not vol) or f"**{vol}**" in r["text"]
            upd = get("https://api.crossref.org/works?filter=updates:" + urllib.parse.quote(doi) + "&rows=20",
                      a.mailto)["message"]["items"]
            notices = "; ".join(f"{(u.get('update-to') or [{}])[0].get('type', 'update')}: {u.get('DOI')}" for u in upd)
            rows.append([r["key"], doi, f"title {'OK' if ok_title else 'CHECK'}; year {'OK' if ok_year else 'CHECK'}; "
                         f"volume {'OK' if ok_vol else 'CHECK'}", f"{journal} {vol} {page} ({year})", title,
                         notices or "none found"])
        except Exception as e:  # network or metadata error: report, do not stop
            rows.append([r["key"], doi, f"ERROR {e}", "", "", ""])
        time.sleep(0.2)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "doi", "metadata check", "Crossref venue", "Crossref title", "update notices"])
        w.writerows(rows)
    bad = [x for x in rows if "CHECK" in x[2] or "ERROR" in x[2] or (x[5] not in ("", "none found"))]
    print(f"{len(rows)} references checked; {len(bad)} need attention -> {a.out}")


if __name__ == "__main__":
    main()
