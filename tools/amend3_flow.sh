#!/bin/bash
# Amendment 3 flow on a copy of the small fixture: alias table, offline EMBER labels, K4 rule, new dry run.
set -u
export PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
PY=/home/claude/a15env/bin/python
CODE=../s10_engine/run_s9.py
rm -rf G3 && mkdir -p G3/work G3/data G3/outputs && cp -r F2/data/I1 G3/data/ && cp F2/work/*_records*.jsonl.gz G3/work/
sed -i 's/,Reline,/,RelineStealer,/' G3/data/I1/malicious_dataset_manifest.csv
echo '{"relinestealer": "reline"}' > G3/outputs/alias_table.json
$PY $CODE --root G3 harmonise > G3_h1.log 2>&1 || tail -5 G3_h1.log
$PY - <<'PY'
import csv, json, zipfile, collections
man = list(csv.DictReader(open("G3/data/I1/malicious_dataset_manifest.csv")))
by = collections.defaultdict(list)
for r in man: by[r["family"]].append(r["sha256"].lower())
lines = []
for k, s in enumerate(by["Agent Tesla"][:40]): lines.append({"sha256": s, "family": "agenttesla", "family_confidence": 0.9})
for k, s in enumerate(by["Amadey"][:40]): lines.append({"sha256": s, "family": "amadey" if k % 2 else "smokeloader", "family_confidence": 0.6})
for s in by["RelineStealer"][:5]: lines.append({"sha256": s, "family": "RedLine", "family_confidence": 0.8})
for d in lines: d.update({"md5": "x", "label": 1, "histogram": [1, 2, 3]})
import os; os.makedirs("G3/data/M", exist_ok=True)
with zipfile.ZipFile("G3/data/M/Win32_train.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("Win32_train/part.jsonl", "".join(json.dumps(d) + "\n" for d in lines))
    z.writestr("Win32_train/other.jsonl", json.dumps({"sha256": "0" * 64, "family": "x"}) + "\n")
open("G3/data/M/REVISION.txt", "w").write("fixture\n")
print("fake EMBER records:", len(lines))
PY
$PY $CODE --root G3 ember-labels --offline > G3_ember.log 2>&1 || tail -8 G3_ember.log
$PY $CODE --root G3 harmonise > G3_h2.log 2>&1 || tail -8 G3_h2.log
$PY $CODE --root G3 lexicon > G3_lex.log 2>&1 || tail -5 G3_lex.log
$PY $CODE --root G3 dryrun > G3_dry.log 2>&1 || tail -15 G3_dry.log
$PY $CODE --root G3 freeze-prep > G3_fp.log 2>&1 || tail -5 G3_fp.log
python3 - <<'PY'
import json
h = json.load(open("G3/outputs/harmonise_counts.json"))
print("alias entries:", h["alias_table_entries"], "| K4_claravy rule:", h["K4_claravy"]["rule"], "| K4_triggered:", h.get("K4_triggered"))
print("per family I1:", {f: v for f, v in h["K4_claravy"]["per_family"]["I1"].items() if v["n"]})
print("origins:", {o: h[f"origin{o}"] for o in (1, 2, 3)})
print("family_sets:", json.load(open("G3/work/family_sets.json")))
d = json.load(open("G3/outputs/dryrun_K5_K6_P1_P4.json"))
print("dryrun:", {k: d[k] for k in ("families", "n", "L", "gap_dry", "gap_ci95", "P1_holds", "design", "mean_recall_within", "n_full_effective", "distinct_predictions")})
print("K5:", d["K5_P4"]); print("K6:", {k: d["K6"][k] for k in ("total_hours", "K6_triggered", "n_train_full_design")})
r = json.load(open("G3/outputs/runinfo_ember-labels.json")); print("ember-labels:", {k: r[k] for k in ("shared_sha256", "archives", "hashes_wanted")})
PY
